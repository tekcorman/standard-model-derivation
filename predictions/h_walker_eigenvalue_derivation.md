# Derivation of h (Hashimoto eigenvalue at the P-point)

**Audit anchor:** Foundational. Conditional on Row 4 (k* = 3), Row 6 (srs identification) of `docs/audits/registers/uniqueness_ledger.md`; theorem-grade per `docs/theorems/theorem_bloch_lift_mu.md` (Ramanujan saturation).

## Abstract

We derive the Hashimoto (non-backtracking walk) eigenvalue at the P-point of the srs Brillouin zone: $h = (\sqrt{3} + i\sqrt{5})/2$. This is the solution of the Ihara-Bass quadratic $h^2 - E_P h + (k^* - 1) = 0$ with the positive-imaginary root selected by the chirality of the srs lattice. The eigenvalue satisfies Ramanujan saturation $|h|^2 = k^* - 1 = 2$. The derivation is a quadratic formula applied to two upstream values.

## Framework axioms invoked

None beyond upstream:
- $k^* = 3$ (from `predictions/k_star.py`)
- $E_P = \sqrt{3}$ (from `predictions/srs_E_at_P.py`)
- srs chirality (proven in `proofs/gauge/srs_rparity_chirality.py`: the space group $I4_132$ has point group $O$ = 432, containing only proper rotations — no inversion, no improper rotations)

## Derivation

### Step 1: Ihara-Bass relation

**Theorem** (Ihara, *J. Math. Soc. Japan* **18**, 12–21, 1966; Bass, *Int. J. Math.* **3**, 717–797, 1992): For a $k$-regular graph, the eigenvalue $E$ of the adjacency matrix and the eigenvalue $h$ of the Hashimoto (edge adjacency / non-backtracking walk) matrix are related by:

$$h^2 - E \cdot h + (k - 1) = 0 \tag{1}$$

This is a standard result in spectral graph theory. See also Terras, *Zeta Functions of Graphs*, Cambridge University Press, 2011, Theorem 3.1.

### Step 2: Quadratic formula

Substituting $E = E_P = \sqrt{k^*} = \sqrt{3}$ and $k = k^* = 3$:

$$h^2 - \sqrt{3} \, h + 2 = 0 \tag{2}$$

By the quadratic formula:

$$h = \frac{\sqrt{3} \pm \sqrt{3 - 8}}{2} = \frac{\sqrt{3} \pm \sqrt{-5}}{2} = \frac{\sqrt{3} \pm i\sqrt{5}}{2} \tag{3}$$

The discriminant is negative ($3 - 4 \cdot 2 = -5$), so both roots are complex conjugates.

This is explicit algebra: quadratic formula applied to equation (2). Every step is arithmetic.

### Step 3: Chirality selects the positive-imaginary root

The two roots are $h = (\sqrt{3} + i\sqrt{5})/2$ and $\bar{h} = (\sqrt{3} - i\sqrt{5})/2$.

The srs lattice has space group $I4_132$ (#214). Its point group is $O$ (Schoenflies notation) = 432 (Hermann-Mauguin), which contains 24 proper rotations and **no improper rotations** (no inversion, no mirror planes, no rotoinversions). This is verified computationally in `proofs/gauge/srs_rparity_chirality.py`: all 24 rotation matrices have determinant $+1$.

The chirality of $I4_132$ (vs its enantiomer $I4_332$) selects one handedness. The positive-imaginary root corresponds to the $I4_132$ handedness:

$$h = \frac{\sqrt{3} + i\sqrt{5}}{2} \tag{4}$$

This is a **discrete choice** (left vs right), not a fit. The enantiomer $I4_332$ would give $\bar{h}$, producing the same physics (all observables depend on $|h|$ or $\arg(h)$ through symmetric combinations).

### Step 4: Ramanujan saturation (self-consistency check)

$$|h|^2 = \frac{3 + 5}{4} = \frac{8}{4} = 2 = k^* - 1 \tag{5}$$

This saturates the Ramanujan bound for $k$-regular graphs (Lubotzky, Phillips & Sarnak, *Combinatorica* **8**, 261–277, 1988): a $k$-regular graph is **Ramanujan** if all non-trivial eigenvalues of the adjacency matrix satisfy $|E| \leq 2\sqrt{k-1}$, equivalently all Hashimoto eigenvalues satisfy $|h|^2 \leq k - 1$.

For srs at the P-point: $|h|^2 = k^* - 1 = 2$ exactly. Saturation. This is a consequence of the srs spectral gap and is not assumed.

## Key derived quantities

From $h = (\sqrt{3} + i\sqrt{5})/2$:

| Quantity | Expression | Value | Used by |
|----------|-----------|-------|---------|
| $|h|^2$ | $k^* - 1$ | 2 | Walk attenuation |
| $|h|$ | $\sqrt{2}$ | 1.4142... | Walk amplitude |
| $\arg(h)$ | $\arctan(\sqrt{5}/\sqrt{3})$ | 52.2388° | PMNS phases |
| $\text{Re}(h)$ | $\sqrt{3}/2$ | 0.8660... | — |
| $\text{Im}(h)$ | $\sqrt{5}/2$ | 1.1180... | — |
| $\text{Im}(h)^2/\text{Re}(h)^2$ | $5/3$ | 1.6667 | Chirality factor ($\alpha_1^{\text{full}}/\alpha_1^{\text{bare}}$) |

## Result

$$\boxed{h = \frac{\sqrt{3} + i\sqrt{5}}{2}}$$

Exact algebraic number. $|h| = \sqrt{2}$, $\arg(h) = \arctan\sqrt{5/3} \approx 52.24°$.

## Comparison with experiment

$h$ is not directly measured. Its components enter physics through:
- PMNS phases: $\alpha_{21} = g \cdot \arg(h) \bmod 360°$, $\delta_{CP} = (g-1) \cdot \arg(h^*) \bmod 360°$
- Chirality factor $5/3 = \text{Im}(h)^2/\text{Re}(h)^2$: enters $\theta_{23}$, $m_\tau$, $\lambda_{\text{Higgs}}$
- Walk attenuation $|h|^2/k^* = 2/3$: the per-step survival probability

Verification is through these downstream predictions.

## Open questions

None. The derivation is the quadratic formula (algebra) applied to the Ihara-Bass relation (Ihara 1966, Bass 1992), with chirality selection by the crystallographic handedness of $I4_132$ (verified computationally). Ramanujan saturation is a self-consistency check, not an assumption.


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

- Bass, H. (1992). The Ihara-Selberg zeta function of a tree lattice. *Int. J. Math.* **3**, 717–797.
- Ihara, Y. (1966). On discrete subgroups of the two by two projective linear group over p-adic fields. *J. Math. Soc. Japan* **18**, 12–21.
- Lubotzky, A., Phillips, R. & Sarnak, P. (1988). Ramanujan graphs. *Combinatorica* **8**, 261–277.
- Terras, A. (2011). *Zeta Functions of Graphs: A Stroll through the Garden*. Cambridge University Press. Theorem 3.1.
