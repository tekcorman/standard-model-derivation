# Theorem: m_ν dark correction Im(h)/|h|² is structurally forced (uniqueness template, parallel to β c=1 closure)

**Status:** THEOREM-GRADE (under specified mechanism + algebraicity meta-theorem).
**Date:** 2026-04-29. Slate header added 2026-05-03.
**Slate:** {A1} (substrate) + A2-T (`theorem_A2_mdl_from_finite_register.md`; supplies MDL Lemma 1 leading-term selection) + Type-4 upstream {`theorem_lattice_coupling_algebraicity.md` (D2 algebraicity meta-theorem); `predictions/uniform_Q_density.py` (Theorem A: uniform ρ_Q on Ramanujan circle); Feshbach-contour mechanism (residue at saddle h)}. ADOPTED-PS supplies the bare scale and is a separate open input.
**Affects:** m_ν2, m_ν3 (dark-correction form). Specifically: the form `(1 + α_1·Im(h)/|h|²)` for the multiplicative dark correction is now derived at theorem grade rather than asserted/post-hoc-selected.

**Note:** ADOPTED-PS (the bare neutrino scale m_{ν3}^bare from PS seesaw) remains a separate open input. This theorem closes the *dark-correction form*, not the bare-scale derivation.

**Predecessors:**
- `theorem_beta_uniqueness_closure.md` (β c=1 closure; same template applied)
- `theorem_lattice_coupling_general.md` (algebraicity meta-theorem)
- `predictions/m_nu2_derivation.md` + `predictions/m_nu3_derivation.md` (existing A−/STRICT-SOLID-CONDITIONAL derivations)
- `predictions/uniform_Q_density.py` + `_derivation.md` (Theorem A: uniform ρ_Q on Ramanujan circle)

## 1. The open structural question

`theorem_dark_correction_mdl.md` flagged a residual ambiguity:

> "**m_ν2/m_ν3 (dark-sector, delocalized amplitude):** F* = sin(arg h)? Or F* = Im(h)/|h|² (Pathway 2 contour-integral form)? The current `predictions/m_nu2.py` uses Im(h)/|h|² (squared 1/|h|² normalization) from the author's separate private derivationa. Whether this is the MDL-cheapest form for neutrino masses, or whether sin(arg h) (linear 1/|h|) would also work, is an open question."

Under MDL bit-cost ranking, sin(arg h) wins at L = 2 bits over Im(h)/|h|² at L = 4 bits. Why does m_ν use the more expensive form?

The structural answer (this theorem): **for self-energy mass²-class corrections, the parity-odd functional is FORCED by the contour-integral mechanism to be Im(1/h) = −Im(h)/|h|², not selected by free MDL minimum.** The self-energy structure determines the form; MDL's role is selecting between equally-valid candidates within that form.

## 2. The argument (uniqueness template, parallel to β closure)

### P1 — Source uniqueness

**Same as β.** The substrate's chirality (h ↔ h* under spatial mirror, srs ↔ srs* enantiomer flip) is the unique source of parity violation in the framework's vacuum. Established by D1 axiom audit.

For m_ν: the dark correction is parity-odd because the substrate's chirality enters via the walker eigenvalue h. Same source as β.

### P2 — Functional uniqueness for self-energy mass²-class observables

This is where m_ν differs from β.

**β (amplitude observable):** photon polarization rotation. Mechanism: Berry phase per leading walker eigenmode. Functional: sin(arg h) = Im(h)/|h| at L = 2 bits (one factor of /|h| for unit-phasor normalization).

**m_ν (mass² observable):** self-energy correction to neutrino mass-squared. Mechanism: Feshbach contour integral over the Ramanujan circle. Specifically:

By Theorem A (`predictions/uniform_Q_density_derivation.md`, theorem-grade), the Q-space spectral density ρ_Q(φ) is uniform on the Ramanujan circle |λ|² = k* − 1 = 2 at MDL optimum (under cosmological-N corrections suppressed by O(√(log N / N)) ~ 10⁻²⁹).

The Feshbach self-energy at energy z, integrated against the uniform measure with pole at the walker eigenvalue h inside the unit disk, gives by the residue theorem:

$$\Sigma(h) = \alpha_1^{\rm bare} \cdot \oint \frac{d\phi}{2\pi} \frac{\rho_Q(\phi)}{z - e^{i\phi}}\bigg|_{z \to h} = \frac{\alpha_1^{\rm bare}}{h}$$

(after evaluating the contour integral with uniform ρ_Q, picking up the residue at z = h).

The parity-odd content of Σ is the imaginary part:

$$\mathrm{Im}\,\Sigma(h) = \mathrm{Im}\!\left(\frac{\alpha_1^{\rm bare}}{h}\right) = \alpha_1^{\rm bare} \cdot \mathrm{Im}\!\left(\frac{1}{h}\right) = \alpha_1^{\rm bare} \cdot \mathrm{Im}\!\left(\frac{\bar h}{|h|^2}\right) = -\alpha_1^{\rm bare} \cdot \frac{\mathrm{Im}(h)}{|h|^2}$$

Therefore:

$$\boxed{\;\big|\mathrm{Im}\,\Sigma(h)\big| = \alpha_1^{\rm bare} \cdot \frac{\mathrm{Im}(h)}{|h|^2} = \alpha_1^{\rm bare} \cdot \frac{\sqrt{5}}{4}\;}$$

The factor /|h|² is FORCED by the residue 1/h (where h sits inside the unit disk and the residue evaluation gives 1/h, whose imaginary part is −Im(h)/|h|²). It is not chosen by MDL minimum; it is determined by the contour-integral structure.

By contrast, sin(arg h) = Im(h)/|h| would arise from a DIFFERENT mechanism (e.g., direct amplitude rotation via Berry phase), which is the relevant mechanism for β but NOT for m_ν.

### P3 — Algebraicity (K-membership)

By the generalized algebraicity meta-theorem (`theorem_lattice_coupling_general.md`):

- Im(h)/|h|² = (√5/2)/2 = √5/4 ∈ K = ℚ(√2, √3, √5). ✓

The form Im(h)/|h|² is in K. By Lemma B (Lindemann), no transcendental (e.g., 1/(16π²)) can replace it.

### Channel-select: Im(h)/|h|² is the unique element of the self-energy channel

(Section heading reframed 2026-05-05 from "MDL bit-cost minimum within self-energy class" per the operator split in `theorem_lattice_coupling_general.md` §2 — the doc body already uses the channel-selection framing, the heading just needed to match.)

Within the self-energy channel (parity-odd functional sourced by Σ(h) = α/h structure), the unique element is Im(h)/|h|² (forced by the mechanism — `channel_select(K, self-energy)` succeeds with a unique K-element). The "linear vs squared" comparison with sin(arg h) is moot because sin(arg h) doesn't arise from this mechanism — it arises from amplitude rotation, which is β's territory (a different operator channel; sin(arg h) is above-waterline for β but does not couple to the self-energy channel).

Coefficient = 1 in the multiplicative correction (1 + α_1 · Im(h)/|h|²) follows from:
- The residue theorem gives the bare self-energy with no additional multiplicative factor.
- No other K-element appears in the calculation.

### Composition: theorem grade

P1 + P2 + P3 jointly establish:

> **m_ν dark correction** = (1 + α_1^bare · Im(h)/|h|²) = (1 + α_1^bare · √5/4)
>
> with the form Im(h)/|h|² **structurally forced** (not post-hoc-selected) by:
> 1. Theorem A (uniform ρ_Q, theorem-grade in repo).
> 2. Residue theorem for the Feshbach self-energy contour integral (standard complex analysis).
> 3. Algebraicity meta-theorem (K-membership confirmed).

## 3. Why this differs from β

| Observable | Mechanism | Functional | Tensor character |
|-----------|-----------|------------|------------------|
| β (cosmic birefringence) | Berry phase / amplitude rotation | sin(arg h) = Im(h)/|h| | dim-1 (rotation angle) |
| m_ν2, m_ν3 (mass²) | Feshbach self-energy contour integral (residue at h inside unit disk) | Im(1/h) = −Im(h)/|h|² | dim-2 (mass²) |

The DIFFERENCE in mechanism (amplitude vs self-energy) explains the DIFFERENCE in normalization (/|h| vs /|h|²). Both forms are MDL-permitted, but the OBSERVABLE TYPE selects which mechanism applies. β is a phase rotation, so amplitude mechanism → sin(arg h). m_ν is a mass² correction, so self-energy mechanism → Im(h)/|h|².

The "linear-vs-squared" rule of `theorem_dark_correction_mdl.md` Lemma 3 (currently CONDITIONAL) is now EXPLAINED: it's not a separate selection rule; it's the consequence of which mechanism the observable's structure picks out.

## 4. Effect on m_ν2 / m_ν3 grade

**Before this closure:**
- `m_nu2.py`: STRICT-SOLID-CONDITIONAL on ADOPTED-PS + ADOPTED-Z3 + I-Feshbach.
- Dark-correction form Im(h)/|h|² flagged as "post-hoc selection" in `theorem_dark_correction_mdl.md`.

**After this closure:**
- I-Feshbach was already closed via A5(b) graduation.
- ADOPTED-Z3 was graduated to "mathematically complete" via R3 (`adoption_register.md` 2026-04-20).
- **Dark-correction form Im(h)/|h|² is now THEOREM-GRADE** (this doc).
- ADOPTED-PS (bare neutrino scale m_{ν3}^bare ≈ 0.048277 eV from PS seesaw) remains the only OPEN input.

**Net status of m_ν2 and m_ν3:** STRICT-SOLID-CONDITIONAL-on-ADOPTED-PS, with the structural form (form of correction, dark-correction coefficient) at theorem grade. This is a genuine status improvement: the linear-vs-squared selection is closed.

## 5. Pattern: uniqueness template applies to multiple observable classes

The β closure (`theorem_beta_uniqueness_closure.md`) and this m_ν closure share argument-shape:

1. Identify mechanism (amplitude rotation vs self-energy).
2. Source uniqueness via substrate chirality.
3. Functional uniqueness FORCED by mechanism (sin(arg h) for amplitudes vs Im(h)/|h|² for self-energy).
4. K-membership via algebraicity meta-theorem.
5. Coefficient = 1 by uniqueness.

**This is the uniqueness template, applied a second time.** Two different observables (β, m_ν), two different mechanisms (Berry phase, Feshbach contour), two different parity-odd functionals (sin(arg h), Im(h)/|h|²) — same template structure, same conclusion (coefficient = 1).

## 6. Pending residuals

**ADOPTED-PS (m_{ν3}^bare from Pati-Salam seesaw):** A-grade, separate from this closure. Closure path: rigorous derivation of M_R = (2/3)^10 · M_GUT and m_t(GUT) from A1+A2-T+A3-T. ~3-5 sessions, research-level.

This is the LAST gap for m_ν2 and m_ν3 to graduate fully to UNIQUE-THEOREM-GRADE.

## 7. Cross-references

- `theorem_beta_uniqueness_closure.md` (β c=1 closure; first uniqueness template application)
- `theorem_lattice_coupling_general.md` (algebraicity meta-theorem; K-membership for m_ν dark correction)
- `theorem_dark_correction_mdl.md` (MDL Lemma 1 + Lemma 3 mass²-vs-amplitude conditional; this closure resolves Lemma 3 for m_ν via mechanism specificity)
- `predictions/m_nu2.py` + `predictions/m_nu2_derivation.md` (m_ν2 prediction; status updated by this closure)
- `predictions/m_nu3.py` + `predictions/m_nu3_derivation.md` (m_ν3 prediction; same)
- `predictions/uniform_Q_density.py` + `_derivation.md` (Theorem A — uniform ρ_Q on Ramanujan circle)
- `predictions/feshbach_exponent_principle.py` (Feshbach exponent principle theorem-grade)
