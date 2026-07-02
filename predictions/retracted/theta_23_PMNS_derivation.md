# θ_23_PMNS — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; see docs/framework_axioms.md), G.1 and G.5 are now DERIVED via CDP 2011 (predictions/observer_hilbert_space.py), but the B6 color-vs-generation retraction remains load-bearing here.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation's Step 1 identifies the physical neutrino mass eigenstates `ν_2` and `ν_3` with the `h` and `h*` eigenspaces of `B(P)` respectively, via "P1 Ramanujan selection + the C_3-charged irreps ω, ω²." Quoting:

> "The physical generations ν_2 and ν_3 are identified (via P1 Ramanujan selection + the C_3-charged irreps ω, ω²) with the h and h* eigenspaces respectively."

Under B6 (`docs/theorem_B6_bridge.md`), the C_3 structure at the srs P-point is the color-Z_3 of SU(3)_c (via Spin(6)≅SU(4)→PS embedding), not a generation label. The `h` and `h*` eigenspaces therefore correspond to color components within one Pati-Salam family, not to the `ν_2` and `ν_3` mass eigenstates of the PMNS matrix. The TBM baseline `θ_23^TBM = 45°` (derived in Step 1 from `|h| = |h*|`) and the Class-2 mass-squared perturbation structure (Step 4) therefore lose their generation-labeled identification.

**Re-derivation target**: Sprint 11 workstream B7.3 (mass operator on C³_gen) and B7.5 (PMNS under C³_gen; see `docs/master_plan.md` §Sprint 11). Under Sprint 11, the atmospheric mixing angle emerges from the mismatch between the neutrino and charged-lepton mass operators on the observer's C³_gen Hilbert space.

**What survives as math**: the spectral identities

$$|h| \;=\; |h^{*}| \;=\; \sqrt{k^{*}-1} \;=\; \sqrt{2}, \qquad \frac{\mathrm{Im}(h)^{2}}{\mathrm{Re}(h)^{2}} \;=\; \frac{5}{3}, \qquad \alpha_{1}^{\mathrm{bare}} \;=\; \left(\frac{k^{*}-1}{k^{*}}\right)^{g-2} \;=\; \left(\frac{2}{3}\right)^{8}$$

are rigorous statements about the srs P-point Hashimoto eigenvalue (`docs/theorem_BP_doubly_degenerate_h.md`) and the Feshbach coupling (`docs/theorem_Feshbach_coupling_strength.md`). The composite

$$\alpha_{1}^{\mathrm{full}} \;=\; \frac{5}{3} \cdot \left(\frac{2}{3}\right)^{8} \;=\; \frac{1280}{19683}$$

is a **standalone color-sector spectral lemma** about the srs Ramanujan circle + Feshbach self-energy. The identity `tan(θ_23) = (1 + α_full)/(1 − α_full)` applied to the color-sector two-state block is an algebraic consequence of 2×2 degenerate perturbation theory (Sakurai §5.2); the result `48.72°` is a valid statement about that color-sector block. Only the identification of this 2×2 block with the atmospheric `ν_2`-`ν_3` generation pair is retracted.

## Specific failing step

Step 1 identifies ν_2 and ν_3 with the h and h* eigenspaces of B(P), labelled by the C_3-charged irreps ω and ω². Under B6, those irreps are color components within one PS family. The subsequent degenerate-perturbation-theory analysis (Step 4–5) operates on a 2×2 color-sector block, not on the ν_2-ν_3 neutrino generation pair. The PMNS atmospheric angle θ_23 is therefore not derived by this chain.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (NuFIT 6.0, NO) | Status |
|---|---|---|---|
| θ_23 | 48.72° | 49.2° ± 1.0° | not explanatory under current framework (within 1σ numerically) |

The numerical match between the color-sector spectral calculation and the observed atmospheric angle is an empirical coincidence under the retracted reading.

## Preserved original derivation (for reference; superseded)

---

# Derivation of $\theta_{23}$ (PMNS atmospheric mixing angle) (SUPERSEDED, retained for reference)

## Abstract

We derive

$$\theta_{23} \;=\; \arctan\!\left(\frac{1 + \alpha_{1}^{\text{full}}}{1 - \alpha_{1}^{\text{full}}}\right), \qquad \alpha_{1}^{\text{full}} \;=\; \frac{\mathrm{Im}(h)^{2}}{\mathrm{Re}(h)^{2}}\,\left(\frac{k^{*}-1}{k^{*}}\right)^{g-2} \;=\; \frac{5}{3}\cdot\left(\tfrac{2}{3}\right)^{8} \;=\; \tfrac{1280}{19683}.$$

Numerically, $\theta_{23} = 48.7207^{\circ}$, against the NuFIT 6.0 observation $49.2 \pm 1.0^{\circ}$: agreement at $0.48\sigma$.

## Framework axioms invoked

Upstream theorems plus Theorem A (`docs/theorem_uniform_Q_density.md`) and Lemma 1 + Exponent Principle (`docs/theorem_Feshbach_coupling_strength.md`).

## Derivation

### Step 1. TBM baseline $\theta_{23}^{\text{TBM}} = 45^{\circ}$ [FAILING STEP under B6]

> The physical generations ν_2 and ν_3 are identified (via P1 Ramanujan selection + the C_3-charged irreps ω, ω²) with the h and h* eigenspaces respectively.

Since $h$ and $h^{*}$ are complex conjugates, $|h| = |h^{*}|$ exactly, hence $m_{\nu_{3}}/m_{\nu_{2}} = 1$ (exact degeneracy at TBM), giving $\theta_{23}^{\text{TBM}} = 45°$.

**This is the failing step under B6**: ν_2 and ν_3 are identified with h and h* via C_3-charged irreps, which B6 proves are color labels.

### Step 2. Feshbach self-energy shape from Theorem A

Under uniform ρ_Q on the Ramanujan circle, $\Sigma(h) = \alpha_1^{\text{bare}}/h$.

### Step 3. Coupling magnitude $\alpha_{1}^{\text{bare}} = (2/3)^{8}$

By the Exponent Principle + Lemma 1: $\alpha_1^{\text{bare}} = ((k^{*}-1)/k^{*})^{g-2} = (2/3)^{8}$.

### Step 4. Class-2 coefficient $\mathrm{Im}(h)^{2} / \mathrm{Re}(h)^{2} = 5/3$

Via $C_{3} \times$ parity selection and 2×2 degenerate perturbation theory:
$$\Delta\theta_{23} \;=\; \tan^{2}(\arg h)\,\alpha_{1}^{\text{bare}} \;=\; \frac{5}{3}\,(2/3)^{8}.$$

### Step 5. Symmetric splitting and corrected mixing angle

$\tan\theta_{23} = (1 + \alpha_1^{\text{full}})/(1 - \alpha_1^{\text{full}})$.

### Step 6. Numerical evaluation

$\alpha_1^{\text{full}} = 0.06503$; $\theta_{23} = \arctan(1.1391) = 48.72°$.

## Result (color-sector spectral lemma only; atmospheric-angle identification retracted under B6)

$$\theta_{23,\text{lemma}} \;=\; 48.7207^{\circ}, \qquad \alpha_{1}^{\text{full}} \;=\; \tfrac{1280}{19683}.$$

## References

- Bass, H. (1992). *Int. J. Math.* **3**, 717–797.
- Feshbach, H. (1958). *Ann. Phys.* **5**, 357–390.
- Sakurai, J.J. & Napolitano, J. (2020). *Modern Quantum Mechanics*, 3rd ed. Cambridge University Press.
- Stein, E.M. & Shakarchi, R. (2003). *Complex Analysis.* Princeton University Press.
- Terras, A. (2011). *Zeta Functions of Graphs.* Cambridge University Press.
- Esteban et al. (2024). NuFIT 6.0.
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
