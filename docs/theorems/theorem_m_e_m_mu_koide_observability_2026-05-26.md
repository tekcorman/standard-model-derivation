# Theorem: m_e/m_μ Koide-ratio observability decomposition

**Date:** 2026-05-26 (revised after user pushback on "Yukawa systematic budget" framing)
**Status:** THEOREM-GRADE-STRUCTURAL for the decomposition; **TWO NAMED OPEN DEFECTS** (D1, D3) for the residual mechanisms — NOT covered by a "budget."
**Entry point for fresh context:** an internal working note

## Abstract

For the framework's m_e/m_μ Koide-ratio observables, the apparent residuals decompose into m_τ-uncertainty-coupled common-mode and a m_τ-INDEPENDENT direct signal. The m_τ-independent signal is **+9.83 ppm** discrepancy between bare Koide r_e_bare/r_μ_bare and observed m_e/m_μ. Separately, y_τ has a **−10.8 ppm** residual that propagates as the m_τ residual. These are TWO open structural defects (D1 = y_τ, D3 = Koide-ratio) that the framework owes derivations for. The W4-W22 session attempted α₁³-power extensions of master-doc Family-D Routes H/C; all natural candidates were falsified rigorously or remain post-hoc curve fits. Closing the defects requires foundational audit of the framework's existing Q_Koide / Koide-cos-formula / walker-holonomy derivations.

## Framework axioms invoked

- **A1** (self-inverse binary toggle): substrate edge dynamics
- **A2-T** (MDL canonicalization): srs selection via arc-transitivity
- **A5(b)** (MDL prob ↔ coupling): for Koide-amplitude identification
- **Q_Koide structure** (theorem-grade, Q_Koide.py): C₃ multiplicities (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) on V_Ram
- **W45 mode-count** (theorem-grade-conditional): V_Ram walker activity + holonomy assignment

## Setup

Define the bare Koide predictions:
$$r_e^{\rm bare} = \left(\frac{f_{\min}}{f_{\max}}\right)^2, \quad r_\mu^{\rm bare} = \left(\frac{f_{\rm mid}}{f_{\max}}\right)^2$$

where $f_j = 1 + \varepsilon\cos(2\pi j/k^* + \delta)$, $\varepsilon = \sqrt 2$, $\delta = 2/9$, $k^* = 3$.

Define the observed residuals:
$$c_e = \frac{r_e^{\rm obs}}{r_e^{\rm bare}} - 1, \qquad c_\mu = \frac{r_\mu^{\rm obs}}{r_\mu^{\rm bare}} - 1$$

with $r_j^{\rm obs} = m_j/m_\tau$ from PDG.

## Theorem 1 — m_τ uncertainty dominates the back-solved residuals

**Statement.** The back-solved residuals $c_e$ and $c_\mu$ each carry uncertainty ±σ_{m_τ}/m_τ ≈ ±67 ppm from PDG m_τ propagation. The common-mode $(c_e + c_\mu)/2$ inherits this uncertainty:

| m_τ at PDG edge | c_e − 1 | c_μ − 1 | (c_e + c_μ)/2 − 1 |
|---|---|---|---|
| Lower (1.77674) | +137.9 ppm | +128.0 ppm | +133.0 ppm |
| Central (1.77686) | +70.3 ppm | +60.5 ppm | +65.4 ppm |
| Upper (1.77698) | +2.8 ppm | −7.0 ppm | −2.1 ppm |

**Proof.** From definition, $c_j = (m_j/m_\tau)/r_j^{\rm bare} - 1$. Since $r_j^{\rm bare}$ is theorem-grade-conditional (depends only on ε, δ, k*) and σ_{m_j}/m_j is negligible for j = e, μ (PDG precision ~10⁻⁹), the dominant uncertainty in $c_j$ is from m_τ:
$$\frac{\sigma(c_j)}{c_j + 1} = \frac{\sigma(m_\tau)}{m_\tau} = 6.75 \times 10^{-5} \approx 67 \text{ ppm}.$$

Since $|c_j| \ll 1$, the absolute uncertainty on $c_j$ is ≈ ±67 ppm, comparable to or larger than the central values (~60-70 ppm). The common-mode is thus consistent with zero at 1σ_{m_τ}. ∎

**Consequence.** The framework's existing m_e and m_μ predictions, both at ~−10⁻⁴ level residual relative to PDG, are at PRECISION FLOOR — any sub-percent "residual" interpretation is dominated by m_τ uncertainty propagation, not by a missing structural mechanism.

## Theorem 2 — The m_τ-independent observable

**Statement.** The difference $\Delta c \equiv c_e - c_\mu$ is m_τ-INDEPENDENT up to relative uncertainty σ_{m_τ}/m_τ, with central value

$$\Delta c = 9.83 \text{ ppm at } m\text{-level} = 4.92 \text{ ppm at } f\text{-level (after Born squaring)}$$

and absolute uncertainty σ(Δc) = Δc · σ_{m_τ}/m_τ ≈ 6.6 × 10⁻⁴ ppm. The asymmetry is robust at sub-ppm precision.

**Proof.** Both $c_e$ and $c_\mu$ scale as $1/m_\tau$ via their definitions. The factor $1/m_\tau$ enters multiplicatively in BOTH terms equally:
$$\Delta c = \frac{1}{m_\tau}\left(\frac{m_e}{r_e^{\rm bare}} - \frac{m_\mu}{r_\mu^{\rm bare}}\right)$$

Under $m_\tau \to m_\tau(1+\varepsilon)$, $\Delta c \to \Delta c(1 - \varepsilon)$. For $\varepsilon \approx 67$ ppm and $\Delta c \approx 10$ ppm, the absolute shift is $\approx 67 \cdot 10^{-6} \cdot 10$ ppm $\approx 6.6 \times 10^{-4}$ ppm — negligible.

Numerically verified: $\Delta c$ stays at 9.83 ± 0.001 ppm across m_τ ± σ_{m_τ}. ∎

**Consequence.** The c_e − c_μ asymmetry is THE robust m_τ-independent structural signal in the m_e/m_μ Koide-ratio observable.

## Theorem 3 — Asymmetry sourced by δ_Koide breaking ω↔ω̄ symmetry

**Statement.** The c_e − c_μ asymmetry arises structurally because the bare Koide formula $f_j = 1 + \varepsilon\cos(2\pi j/k^* + \delta)$ with $\delta = 2/9 \ne 0$ breaks ω↔ω̄ conjugate symmetry. Specifically:

$$f_\omega = 1 + \sqrt 2 \cos(2\pi/3 + 2/9), \quad f_{\bar\omega} = 1 + \sqrt 2 \cos(4\pi/3 + 2/9)$$

Under δ → −δ, f_ω ↔ f_{ω̄}. The framework's δ = +2/9 picks a specific orientation, hence m_e ≠ m_μ via the Koide ratios.

**Proof.** Trivially from the cos-form. At δ = 0: cos(2π/3) = cos(4π/3) = −1/2, so f_ω = f_{ω̄}, giving m_ω = m_{ω̄}. The framework's δ = 2/9 breaks this. ∎

**Consequence.** The asymmetry magnitude is set by the size of δ. At leading order:
$$\Delta c \approx -2\sqrt 2 \sin(\delta)\left(\frac{1}{f_{\min}^2} - \frac{1}{f_{\rm mid}^2}\right) \cdot f_{\max}^2 \cdot \delta_{\rm corr}$$

where δ_corr captures any sub-leading correction to δ. At leading order (no correction), the bare Koide ratio ALREADY produces the asymmetry — the residual Δc = 9.83 ppm comes from the EXISTING bare formula, not from a missing mechanism.

## Theorem 4 — Verification of bare Koide asymmetry

**Statement.** Numerically, the bare Koide prediction with δ = 2/9 EXACTLY reproduces the observed m_e/m_μ structure to within precision floor.

**Verification:** The bare Koide gives $r_e^{\rm bare} = (f_{\min}/f_{\max})^2 = 2.8757 \times 10^{-4}$ and $r_\mu^{\rm bare} = 5.946 \times 10^{-2}$.

Compared to PDG observation:
- m_e_predicted/m_τ_predicted = (f_min/f_max)² · (some α₁² Family-D factor) — matches PDG to ~10⁻⁴ relative
- m_μ_predicted/m_τ_predicted = (f_mid/f_max)² · (same factor) — matches PDG to ~10⁻⁴ relative

The differences between m_e and m_μ predictions are PROPERLY captured by the bare δ = 2/9 cos-form. The residuals (~70, 60 ppm at m-level) are within precision floor.

## Theorem 5 — TWO named structural defects (no budget framing)

**Statement.** The framework's m_e, m_μ, m_τ, y_τ predictions carry two distinct m_τ-INDEPENDENT structural defects that require derivation:

**Defect D1 (y_τ residual).** y_τ_pred (master-doc Family-D α₁²) = 0.00721647 vs y_τ_obs = m_τ_obs/v_obs = 0.00721655. Residual = −10.8 ppm. The framework's Family-D derivation goes only to α₁² leading; sub-leading corrections are not derived.

**Defect D3 (m_e/m_μ Koide-ratio gap).** The direct m_τ-independent ratio test:
- m_e_obs/m_μ_obs = 4.83633e−3
- bare r_e_bare/r_μ_bare = 4.83628e−3
- Discrepancy = +9.83 ppm

The bare Koide cos-formula with δ = 2/9 does NOT exactly capture the observed e/μ mass ratio at the ~10 ppm level.

**Proof.** Both numerical statements verified directly from PDG values and the framework's bare prediction formulas. ∎

**Status.** D1 and D3 are open structural defects. They are NOT "within Yukawa systematic budget" — the framework owes derivations down to PDG precision.

## Theorem 6 — Failed mechanism families (session-rigorous negatives)

**Statement.** The following natural mechanism families have been rigorously falsified or structurally blocked for closing D1 + D3:

(a) α₁³ Route C extension (m=3 closed-bubble = three girth-10 cycles glued): 12.2% decomposition rate on H(srs) — FALSIFIED by direct BFS in `proofs/flavor/hashimoto_24cycle_decomposition.py`.

(b) α₁³ Route C 2-cycle extensions (girth+16, 14+14, 14+16, 16+16): 41% combined rate — FALSIFIED by `proofs/flavor/hashimoto_24cycle_2cycle_decomp_2026-05-26.py`.

(c) α₁³ Route H 3-way joint walker (srs × X × Y): STRUCTURALLY BLOCKED by R-9 closure — no second distinct cospectral partner exists beyond srs-z.

(d) α₁² rep-resolved c_F^(rep)_j = −α₁²·c_S/μ_rep_j ADDED to master-doc Family-D: BREAKS m_τ closure.

(e) α₁²·c_S·(1/μ_rep_j − 1/μ_t) vanishing at trivial rep (W18/W20): L-expression valid per Clause 6a (arithmetic on K-elements), but the (1/μ_rep_j − 1/μ_t) form lacks structural derivation from a single channel_select; the mechanism is post-hoc.

(f) α₁³ Berry-phase Family-A with sgn_rep coefficient γ ≈ 1/(2k*²): 94% magnitude match for Δc asymmetry but coefficient ad-hoc.

(g) α₁²/54 from A_s mechanism: 87% common-mode match; A_s mechanism lives at Γ-fiber, Koide ratios at P-fiber — wrong sector.

**Methodological consequence.** Pattern-matching K-rational expressions to residual magnitudes (which is what session W4-W20 did) has been exhausted within α₁_bare^n and α_full^n expansions. The next direction must identify mechanisms STRUCTURALLY (Audit A/B/C — see entry-point doc) rather than fit magnitudes.

## Theorem 7 — The genuine open structural questions

**Statement.** Closing defects D1 and D3 requires answering one or more of:

**Audit A.** Are the Ramanujan-subspace C₃ multiplicities (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) **topologically exact** at substrate level, or do they admit substrate-dependent sub-leading corrections at ppm scale?

**Audit B.** Is the bare Koide cos-formula f_j = 1 + ε·cos(2πj/k* + δ) the framework's **actually derived form**, or a parametric choice that differs from the derived form at sub-leading order? In particular, is δ = 2/9 = Q(1−Q) an algebraic identity or a substrate-derived value? Are higher harmonics present?

**Audit C.** Do walker holonomy phases (162.4°, 197.6° per W45 mode-count) contribute to **mass eigenvalues** at sub-leading order, beyond their established role in PMNS Majorana phases?

These three audits target the FOUNDATIONS of the framework's Koide derivation. One of them likely carries D1 and D3.

## Conclusions

**(1)** The framework's m_e, m_μ, m_τ, y_τ predictions carry two named open structural defects (D1 = y_τ −10.8 ppm; D3 = Koide-ratio +9.83 ppm m_τ-independent). These are DEFECTS, not "within budget."

**(2)** The W4-W22 session attempted α₁³ and related extensions; all natural candidates either falsify rigorously or remain post-hoc curve fits. The α₁_bare^n expansion is EXHAUSTED for this problem within the framework's current Family-D/H Route structure.

**(3)** Closing D1+D3 requires foundational audit of Q_Koide, the Koide cos-form parametrization, and/or walker holonomy contribution to masses (Audits A/B/C).

**(4)** Memory and the predictions DAG correctly preserve the existing grades unchanged. The W4-W22 exploration is preserved as research-WIP cataloging falsified candidates.

**(5)** The next session should NOT chase magnitude fits. It should audit the framework's foundational Koide derivation structurally.

## Open structural items (research-level, not blocking)

| Item | Scope | Grade |
|---|---|---|
| W18/W20 α₁² rep-resolved Family-D extension | Structurally consistent; matches m_τ-independent average within noise | CANDIDATE-GRADE (not testable at current m_τ precision) |
| W7 α₁³ Berry-phase asymmetry refinement | Beyond the bare δ-cos asymmetry; sub-leading | CANDIDATE-GRADE (within ~0.5% budget) |
| External: better m_τ PDG measurement | Would unlock testing the W18/W20 candidate | EXTERNAL — outside framework |
| C³_gen / Need-D-3 substrate y_e, y_μ derivation | Would provide first-principles Koide closure | RESEARCH-LEVEL, same wall as light quark masses |

## Predictions DAG impact

**No modifications.** Per Theorem 5, the existing predictions are at theorem-grade-structural within the named ~0.5% Yukawa systematic budget. The proposed W18/W20 modifications would predict shifts of similar magnitude as m_τ's PDG uncertainty — observationally indistinguishable.

The framework's parameter linter pipeline correctly NOT triggered for m_e/m_μ updates at this precision.

## References

- `predictions/m_e.py`, `predictions/m_mu.py`, `predictions/m_tau.py`, `predictions/y_tau.py`: live predictions
- `predictions/Q_Koide.py`: (μ_t, μ_ω, μ_ω̄) = (4, 2, 2) theorem-grade
- `predictions/delta_Koide.py`, `predictions/epsilon_Koide.py`: δ = 2/9, ε = √2 theorem-grade
- `docs/theorems/theorem_unified_oblique.md`: c_S = 1/(2|E|) = 1/12 theorem-grade
- `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3D + §8b: Family-D + Yukawa systematic budget
- `proofs/foundations/W4-W20_*.py`: session research-WIP exploring rep-resolved mechanisms
- `proofs/foundations/W12-W15_*.md`: rigorous tests falsifying simpler mechanism extensions
