# m_ν2 — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; docs/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 (predictions/observer_hilbert_space.py), but the B6 color-vs-generation retraction, Feshbach/uniform Q-density formalism, and Pati-Salam labeling remain separately load-bearing.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation identifies the Pati-Salam seesaw second-generation slot `m_ν2` with the h-eigenspace Class-1 Feshbach amplitude correction, via the Ihara splitting ratio `R = 228/7` and the identification of the C_3-charged irreps at the P-point with neutrino mass eigenstates. Specifically:

- Step 1 uses `m_ν2_bare = m_ν3_bare / √R` with `R = 228/7` (the Ihara splitting), where the `ν_2` slot is indexed by a generation label.
- Step 3 identifies the Class-1 amplitude coefficient `|Im Σ(h)| = (√5/4) α_1^bare` with a multiplicative correction to the neutrino generation-2 mass.

Under B6 (`docs/theorem_B6_bridge.md`), the C_3 structure at the srs P-point is color-Z_3 of SU(3)_c, not a generation label. The h-eigenspace is a color-sector sub-bundle within one Pati-Salam family, not a neutrino mass eigenstate indexed by generation. The Class-1 coefficient `√5/4` is a valid spectral shape factor (= Im(h)/|h|²) but its identification with a generation-2 neutrino correction relies on the retired C_3-as-generation reading.

**Re-derivation target**: Sprint 11 workstream B7.3 (mass operator on C³_gen; see `docs/master_plan.md` §Sprint 11). Under Sprint 11, neutrino masses are eigenvalues of a 3×3 Hermitian mass operator on C³_gen, which is orthogonal to the srs C_3 color structure.

**What survives as math**: the closed-form spectral identity

$$\left|\mathrm{Im}\,\frac{1}{h}\right| \;=\; \frac{\mathrm{Im}(h)}{|h|^{2}} \;=\; \frac{\sqrt{5}/2}{2} \;=\; \frac{\sqrt{5}}{4}$$

for the srs P-point eigenvalue `h = (√3 + i√5)/2` (Ramanujan-saturated, |h|² = 2), and

$$\alpha_{1}^{\mathrm{bare}} \;=\; \left(\frac{k^{*}-1}{k^{*}}\right)^{g-2} \;=\; \left(\frac{2}{3}\right)^{8}$$

(Lemma 1 + Exponent Principle, `docs/theorem_Feshbach_coupling_strength.md`), together give the **standalone color-sector spectral lemma**

$$c_{1}\,\alpha_{1}^{\mathrm{bare}} \;=\; \frac{\sqrt{5}}{4}\,\left(\frac{2}{3}\right)^{8} \;=\; 0.02181\ldots$$

This is label-agnostic and remains valid. Only the identification of this number with a multiplicative correction to the generation-2 neutrino mass is retracted.

## Specific failing step

Step 1 (bare scale) writes `m_ν2_bare = m_ν3_bare / √R`, which requires a generational slot structure. Step 3 (Class-1 amplitude coefficient) identifies the shape factor `Im(h)/|h|² = √5/4` as the correction to a *generation-2* neutrino mass. Both steps depend on reading the C_3 structure at the P-point as a three-generation label.

Additionally, the bare scale `m_ν3_bare = m_t(GUT)²/M_R` was already A-grade (not theorem-grade) under the previous rigor classification, so this derivation was never fully theorem-grade even before B6.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (NuFIT 6.0) | Status |
|---|---|---|---|
| m_ν2 | 8.644 meV | 8.654 ± 0.110 meV | not explanatory under current framework (within 1σ numerically) |

The numerical match is an empirical coincidence under the retracted reading, further driven by the A-grade bare seesaw pipeline. Whether it is re-derivable under the C³_gen mass operator is the Sprint 11 B7.3 open question.

## Preserved original derivation (for reference; superseded)

---

# Derivation of the second light-neutrino mass $m_{\nu_2}$ (SUPERSEDED, retained for reference)

## Abstract

We derive
$$m_{\nu_2} \;=\; m_{\nu_2}^{\mathrm{bare}}\,\left(1 + \tfrac{\sqrt 5}{4}\,\alpha_1^{\mathrm{bare}}\right) \;=\; 8.644\ \mathrm{meV}$$
from: (i) Pati-Salam seesaw bare scale `m_ν2_bare = m_ν3_bare / √R`, (ii) Ihara splitting `R = 228/7`, (iii) Class-1 Feshbach amplitude correction with coefficient `√5/4 · (2/3)⁸` from Theorem A + Exponent Principle.

## Framework axioms and theorems invoked

- A1, A2; `theorem_walker_dynamics.md`; `theorem_BP_doubly_degenerate_h.md`; Theorem A (`theorem_uniform_Q_density.md`); Lemma 1 + Exponent Principle (`theorem_Feshbach_coupling_strength.md`); `predictions/R_nu_splitting.py`.

## Setup: Type D Class 1

By `docs/W4_identification_catalog.md` §2D, m_ν2, m_ν3 are Type-D Class-1 observables: multiplicative amplitude dark correction `O = O^bare (1 + c_1 α_1^bare)` with c_1 derived from the P-point Feshbach self-energy.

## Derivation

### Step 1. Bare scale via Pati-Salam seesaw and Ihara splitting [FAILING STEP under B6]

From `proofs/masses/srs_nu_mass_ps.py`:
$$m_{\nu_3}^{\mathrm{bare}} \;=\; \frac{m_t(\mathrm{GUT})^{2}}{M_R}, \qquad M_R \;=\; (2/3)^{10}\,M_{\mathrm{GUT}},$$
giving `m_ν3_bare ≈ 0.04828 eV`. With `R = 228/7` and `m_ν1 = 0`:
$$m_{\nu_2}^{\mathrm{bare}} \;=\; m_{\nu_3}^{\mathrm{bare}} / \sqrt{R} \;\approx\; 0.008459\ \mathrm{eV}.$$

**Failing step under B6**: the generational slot indexing (ν_2 = h-eigenspace, ν_3 = a different h-related eigenspace of the seesaw) depends on C_3-as-generation.

### Step 2. Feshbach self-energy under Theorem A

Under uniform ρ_Q: $\Sigma(h) = \alpha_1^{\text{bare}}/h$.

### Step 3. Class-1 amplitude coefficient $\sqrt 5/4$

$\mathrm{Im}(1/h) = -b/|h|^2 = -\sqrt{5}/4$, so $|\mathrm{Im}\,\Sigma(h)| = \alpha_1^{\text{bare}} \cdot \sqrt{5}/4$.

### Step 4. Exponent Principle

$\alpha_1^{\text{bare}} = (2/3)^{g-2} = (2/3)^8$.

### Step 5. Combine

$m_{\nu_2} = 0.008459 \times 1.02181 = 8.644$ meV.

## Result (color-sector spectral lemma only; generation-2 neutrino identification retracted under B6)

$$m_{\nu_2, \text{lemma}} \;=\; m_{\nu_2}^{\mathrm{bare}}\,(1 + \tfrac{\sqrt 5}{4}\,\alpha_{1}^{\mathrm{bare}}) \;=\; 8.644\ \mathrm{meV}.$$

## References

- Bass, H. (1992). *Int. J. Math.* **3**, 717–797.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley-Interscience.
- Esteban et al. (2024). NuFIT 6.0.
- Pati, J.C. & Salam, A. (1974). *Phys. Rev. D* **10**, 275.
- Sunada, T. (2012). *Topological Crystallography*.
- Terras, A. (2011). *Zeta Functions of Graphs*.
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
