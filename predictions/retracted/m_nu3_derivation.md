# m_ν3 — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. Under the three-axiom framework (A1+A2+A3; docs/framework_axioms.md), G.1 and G.5 are DERIVED via CDP 2011 (predictions/observer_hilbert_space.py), but the B6 color-vs-generation retraction, Feshbach/uniform Q-density formalism, and Pati-Salam labeling remain separately load-bearing.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation identifies the Pati-Salam seesaw third-generation slot `m_ν3` with the h-eigenspace Class-1 Feshbach amplitude correction. Specifically:

- Step 1 uses `m_ν3_bare = m_t(GUT)²/M_R` with the "third-generation slot" identification `m_{u_3} = m_t`, where `j = 3` is a generation index.
- Step 3 identifies the Class-1 amplitude coefficient `|Im Σ(h)| = (√5/4) α_1^bare` with a multiplicative correction to the neutrino generation-3 mass.

Under B6 (`docs/theorem_B6_bridge.md`), the C_3 structure at the srs P-point is color-Z_3 of SU(3)_c, not a generation label. The h-eigenspace is a color-sector sub-bundle within one Pati-Salam family, not a neutrino mass eigenstate indexed by generation. The "third-generation slot" and the Class-1 coefficient's identification with a generation-3 correction both rely on the retired C_3-as-generation reading.

**Re-derivation target**: Sprint 11 workstream B7.3 (mass operator on C³_gen; see `docs/master_plan.md` §Sprint 11).

**What survives as math**: same spectral lemma as `m_nu2_derivation.md`:

$$\left|\mathrm{Im}\,\frac{1}{h}\right| \;=\; \frac{\sqrt{5}}{4}, \qquad \alpha_{1}^{\mathrm{bare}} \;=\; \left(\frac{2}{3}\right)^{8}, \qquad c_{1}\,\alpha_{1}^{\mathrm{bare}} \;=\; \frac{\sqrt{5}}{4}\,\left(\frac{2}{3}\right)^{8} \;=\; 0.02181\ldots$$

is a **standalone color-sector spectral lemma** about the srs P-point Hashimoto eigenvalue and Feshbach coupling. Label-agnostic, remains valid under B6. Only the identification with a generation-3 neutrino correction is retracted.

## Specific failing step

Step 1 (bare scale) identifies `m_ν3_bare` with the third-generation slot of the Pati-Salam seesaw via `m_{u_3} = m_t`. Step 3 (Class-1 coefficient) identifies `Im(h)/|h|² = √5/4` as the correction to the generation-3 neutrino mass. Both steps require a generational indexing on the h-eigenspace spectral data; B6 retires this reading.

Additionally, the bare scale was already A-grade (not theorem-grade) under the previous rigor classification, so this derivation was never fully theorem-grade even before B6.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (NuFIT 6.0) | Status |
|---|---|---|---|
| m_ν3 | 49.33 meV | 50.34 ± 0.24 meV | not explanatory under current framework (~4.2σ tension, driven by A- bare scale) |

The 4.2σ tension is driven entirely by the A-grade bare seesaw pipeline; the Class-1 multiplicative factor is a color-sector spectral calculation whose identification with a neutrino-generation correction is retracted.

## Preserved original derivation (for reference; superseded)

---

# Derivation of the heaviest light-neutrino mass $m_{\nu_3}$ (SUPERSEDED, retained for reference)

## Abstract

$$m_{\nu_3} \;=\; m_{\nu_3}^{\mathrm{bare}}\,\left(1 + \tfrac{\sqrt 5}{4}\,\alpha_1^{\mathrm{bare}}\right) \;=\; 49.33\ \mathrm{meV}.$$

Sister of `m_nu2_derivation.md`: same theorems, same Class-1 coefficient, same coupling. The only difference is the generation slot — `m_ν3` takes `m_t(GUT)²/M_R` directly.

## Framework axioms and theorems invoked

Identical to `m_nu2_derivation.md`.

## Derivation

### Step 1. Bare scale via Pati-Salam seesaw [FAILING STEP under B6]

From Cl(6) → SU(4)_PS, `M_D(ν) = M_u^T`. In the mass eigenbasis at GUT:
$$m_{\nu_j}^{\mathrm{bare}} \;=\; \frac{m_{u_j}(\mathrm{GUT})^{2}}{M_R}, \quad j \in \{1,2,3\}.$$

Third-generation slot (j = 3): $m_{u_3} = m_t$, so
$$m_{\nu_3}^{\mathrm{bare}} \;=\; \frac{m_t(\mathrm{GUT})^{2}}{M_R} \;\approx\; 0.04828\ \mathrm{eV}.$$

**Failing step under B6**: the generational slot indexing depends on C_3-as-generation.

### Step 2. Feshbach self-energy under Theorem A

$\Sigma(h) = \alpha_1^{\text{bare}}/h$.

### Step 3. Class-1 amplitude coefficient $\sqrt 5/4$

$|\mathrm{Im}\,\Sigma(h)| = \alpha_1^{\text{bare}} \cdot \sqrt{5}/4$.

### Step 4. Exponent Principle

$\alpha_1^{\text{bare}} = (2/3)^8$.

### Step 5. Combine

$m_{\nu_3} = 0.04828 \times 1.02181 = 49.33$ meV.

## Result (color-sector spectral lemma only; generation-3 neutrino identification retracted under B6)

$$m_{\nu_3, \text{lemma}} \;=\; m_{\nu_3}^{\mathrm{bare}}\,(1 + \tfrac{\sqrt 5}{4}\,\alpha_{1}^{\mathrm{bare}}) \;=\; 49.33\ \mathrm{meV}.$$

## Consistency with $m_{\nu_2}$

Because both `m_ν2` and `m_ν3` inherit the same Class-1 factor, the ratio
$$\frac{m_{\nu_3}^{2}}{m_{\nu_2}^{2}} \;=\; R \;=\; \frac{228}{7}$$
holds exactly — the Class-1 correction cancels. This algebraic consistency is preserved under B6 as a statement about the same color-sector spectral block for the two retracted generation labels.

## References

- Bass, H. (1992). *Int. J. Math.* **3**, 717–797.
- Esteban et al. (2024). NuFIT 6.0.
- Feshbach, H. (1958). *Ann. Phys.* **5**, 357–390.
- Pati, J.C. & Salam, A. (1974). *Phys. Rev. D* **10**, 275.
- Sunada, T. (2012). *Topological Crystallography*.
- Terras, A. (2011). *Zeta Functions of Graphs*.
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
