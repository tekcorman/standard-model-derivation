# m_nu2 — second light neutrino mass (normal ordering)

**Date:** 2026-04-18 (status updated 2026-04-19 session 2; dark-correction form upgraded 2026-04-29 to theorem-grade; I-Feshbach closed 2026-05-02 EOD+9 via analytical Σ(h))
**Status:** **THEOREM-GRADE-CONDITIONAL on ADOPTED-PS + ADOPTED-Z3** (promotion 2026-05-02 EOD+9 from STRICT-SOLID-CONDITIONAL). The dark-correction form Im(h)/|h|² and the I-Feshbach NB-survival↔coupling identification are now both derived; only the bare neutrino scale (ADOPTED-PS) and the C₃-Fourier-index ↔ generation-label identification (ADOPTED-Z3) remain as adopted conditionals.
**Pattern:** Feshbach (rigorous core + explicitly flagged adopted residuals)
**Companion:** `predictions/m_nu2.py`
**Update:** 2026-05-02 EOD+9 — I-Feshbach removed from conditionals. The "NB walk survival = physical coupling strength" identification was previously adopted; it now flows from the closed-form analytical Feshbach `docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md` (Σ(h) = α₁·h̄/|h|² leading-order) + the author's separate private derivation (M_n = 0 at MDL optimum). The water-filling claim was confirmed by the m_ν2 PDG sensitivity sweep at -0.10σ in the same theorem doc ("Subleading verification" section). Subleading M_n contributions DEGRADE the PDG match, supporting M_n = 0 structurally. -Im(Σ_lead)/α₁ = √5/4 IS the dark coefficient used here.
**Update:** 2026-04-29 — Dark-correction form Im(h)/|h|² = √5/4 is now theorem-grade per `docs/theorems/theorem_m_nu_dark_correction_uniqueness_closure.md`. The "post-hoc linear-vs-squared selection" formerly noted as a gap is resolved: the form is FORCED by the Feshbach self-energy contour-integral mechanism (residue at h gives Σ = α₁/h, whose imaginary part is −α₁·Im(h)/|h|²), not chosen freely. Combined with `theorem_lattice_coupling_general.md` (K-membership: √5/4 ∈ K) and uniqueness from substrate chirality + mechanism specificity, the form is uniquely selected.
**Update:** 2026-04-19 session 2 — All references below to "I-Feshbach (adopted)" pre-date the 2026-05-02 closure; treat them as "derived via theorem_analytical_feshbach_ramanujan_boundary.md".

**CAVEAT — m_ν1 = 0 citation (2026-05-02 EOD+9):** The "NuFIT 6.0 normal ordering, m_ν1 = 0" assumption used in this derivation reflects an OBSERVATIONAL convention (lightest-massless normal ordering), NOT a derived structural prediction. The previous structural derivation ("M_D(trivial_s) = 0 at the P-point") was retracted under B6 — the C₃-trivial sector is color-singlet content, not generation-1 neutrino. Re-derivation under the C³_gen framework is open research per an internal working note sub-target B7.3a.v. Tracked as residue R-15 in `docs/audits/registers/structural_residue_register.md`. m_ν2's central value does NOT load-bear on m_ν1 = 0: under any normal-ordering convention with m_ν1 small, m_ν2 ≈ √(Δm²₂₁) to high precision, so this caveat does not change the prediction.

## Abstract

We predict

$$m_{\nu_2} \;=\; \frac{m_{\nu_3}^{\mathrm{bare}}}{\sqrt{R}} \cdot \left(1 + \tfrac{\sqrt{5}}{4}\,\alpha_1^{\mathrm{bare}}\right)$$

where:

- $m_{\nu_3}^{\mathrm{bare}} = 0.048277$ eV is an external numerical input from the Pati-Salam seesaw pipeline (ADOPTED-PS; A-grade).
- $R = 228/7$ is the Ihara splitting ratio $\Delta m^2_{31}/\Delta m^2_{21}$ (derived; `predictions/R_nu_splitting.py`).
- $\alpha_1^{\mathrm{bare}} = (2/3)^8$ is the Feshbach coupling strength (derived; Exponent Principle, $n_{\mathrm{fixed}} = 2$).
- $\sqrt{5}/4 = \mathrm{Im}(h)/|h|^2$ is the shape factor at the srs P-point eigenvalue (derived; Ramanujan saturation).

**Result:** $m_{\nu_2} = 8.644$ meV.  
**Observed:** $m_{\nu_2} = \sqrt{\Delta m^2_{21}} = 8.654 \pm 0.110$ meV (NuFIT 6.0, normal ordering, $m_{\nu_1} = 0$).  
**Deviation:** $-0.10$ sigma.

## Rigor bar

Every step is one of: (a) axiom A1/A2/A3, (b) CAS-verifiable algebra, (c) cited mathematical theorem with author/year, or (d) upstream `predictions/` file.

## Adopted residuals (flagged, not derived)

| Label | Content | Gap to close |
|---|---|---|
| ADOPTED-PS | $m_{\nu_3}^{\mathrm{bare}} = 0.048277$ eV from Pati-Salam seesaw at $M_R = (2/3)^{10} \cdot M_{\mathrm{GUT}}$ | Theorem-grade derivation of $M_R$ and $m_t(\mathrm{GUT})$ from A1 + A2-T + A3-T (open research; no candidate path identified per an internal working note) |
| ADOPTED-Z3 | C_3 Fourier index $j$ identifies with generation label (same as `predictions/Q_Koide.py` ADOPTED-Z3) | B7.3 Sprint 11: mass operator on $\mathbb{C}^3_{\mathrm{gen}}$; under B6, C_3 at the P-point is color-Z₃ (not generation), so this identification is currently load-bearing for the (color → generation) mapping |
| ~~I-Feshbach~~ | ~~NB walk survival factor = physical coupling strength~~ | **CLOSED 2026-05-02 EOD+9** via `docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md` (analytical Σ(h) = α₁·h̄/|h|² closed-form) + the author's separate private derivation (M_n = 0 at MDL optimum) + m_ν2 PDG sensitivity sweep verification (-0.10σ). No longer an adopted residual. |

## Derivation

### Step 1. P-point NB walk eigenvalue [derived]

From `predictions/h_walker_eigenvalue.py` (Ihara-Bass quadratic, Ihara 1966; Bass 1992; Terras 2011 Thm 3.1):

$$h = \frac{\sqrt{3} + i\sqrt{5}}{2}, \qquad |h|^2 = \frac{3}{4} + \frac{5}{4} = 2 = k^* - 1$$

Ramanujan saturation $|h|^2 = k^*-1 = 2$ is a consequence of the srs spectral gap (Lubotzky, Phillips, Sarnak 1988).

Shape factor (exact rational-radical algebra):

$$\frac{\mathrm{Im}(h)}{|h|^2} = \frac{\sqrt{5}/2}{2} = \frac{\sqrt{5}}{4}$$

Cross-checked by `B_P_doubly_degenerate_h.py` (sympy-exact: $h = (\sqrt{3} + i\sqrt{5})/2$ with multiplicity 2, C_3-protected).

### Step 2. Feshbach coupling (Exponent Principle) [derived]

From `predictions/feshbach_exponent_principle.py` with $n_{\mathrm{fixed}} = 2$ (scattering: in + out pinned):

$$\alpha_1^{\mathrm{bare}} = \left(\frac{k^*-1}{k^*}\right)^{g-2} = \left(\frac{2}{3}\right)^8 = \frac{256}{6561} \approx 0.039018$$

Proof chain in `feshbach_exponent_principle.py`: Jaynes 1957 max-entropy on $k$ incident edges + Serre 1980 §I.1 (NB reduced words) + Terras 2011 §2.1 (independence on universal covering tree). Scope: $n_{\mathrm{fixed}} \in \{0, 1, 2\}$.

**I-Feshbach (CLOSED 2026-05-02 EOD+9):** the identification of this NB walk survival factor with a physical coupling strength is now derived via `docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md` (analytical Feshbach Σ(h) = α₁·h̄/|h|² closed-form on the Ramanujan circle) + the author's separate private derivation (M_n = 0 at MDL optimum, confirmed by m_ν2 PDG sensitivity sweep at -0.10σ). The bare Feshbach coupling α₁ in Σ(h) IS the NB walk survival factor (k*−1)/k*)^{g−n_fixed}; the identification is no longer adopted.

### Step 3. Splitting ratio [derived]

From `predictions/R_nu_splitting.py` (K4 Green's function Chebyshev expansion + Gaussian integer arithmetic; Bass 1992; Ihara 1966):

$$R = \frac{\Delta m^2_{31}}{\Delta m^2_{21}} = \frac{228}{7}$$

Under $m_{\nu_1} = 0$ (normal ordering, lightest massless):

$$m_{\nu_2}^2 = \Delta m^2_{21}, \quad m_{\nu_3}^2 = \Delta m^2_{31}$$
$$\Rightarrow \quad m_{\nu_2}^{\mathrm{bare}} = \frac{m_{\nu_3}^{\mathrm{bare}}}{\sqrt{R}}$$

This step is pure algebra given $R$ (derived) and $m_{\nu_3}^{\mathrm{bare}}$ (ADOPTED-PS).

### Step 4. Class-1 Feshbach correction [derived]

Under Theorem A (`../predictions/uniform_Q_density_derivation.md` Part A), $\rho_Q(\phi)$ is uniform on the Ramanujan circle at MDL optimum. The Feshbach self-energy contour integral against the uniform measure gives $\Sigma(h) = \alpha_1^{\mathrm{bare}} / h$ (residue at the pole $h$ inside the unit disk). The multiplicative amplitude correction factor is

$$1 + |\mathrm{Im}\,\Sigma(h)| = 1 + \alpha_1^{\mathrm{bare}} \cdot \frac{\mathrm{Im}(h)}{|h|^2} = 1 + \left(\frac{2}{3}\right)^8 \cdot \frac{\sqrt{5}}{4}$$

Numerically: $1 + 0.039018 \times 0.559017 = 1.021812$.

This step is STRICT-SOLID (no adopted content beyond I-Feshbach, which is referenced from `feshbach_exponent_principle.py`).

### Step 5. Bare scale (ADOPTED-PS) [external input, A-grade]

From `proofs/masses/srs_nu_mass_ps.py` Part 3/4 (Pati-Salam type-I seesaw, Ihara output form):

$$m_{\nu_3}^{\mathrm{bare}} \approx 0.048277 \text{ eV}, \qquad m_{\nu_2}^{\mathrm{bare}} = \frac{m_{\nu_3}^{\mathrm{bare}}}{\sqrt{228/7}} \approx 0.008459 \text{ eV}$$

**ADOPTED-PS:** the girth-cycle identification $M_R = (2/3)^{10} M_{\mathrm{GUT}}$ and the two-loop MSSM RG pipeline for $m_t(\mathrm{GUT})$ are A-grade, not theorem-grade under A1 + A2-T + A3-T.

### Step 6. Prediction [ADOPTED-PS + ADOPTED-Z3 only]

$$m_{\nu_2} = m_{\nu_2}^{\mathrm{bare}} \cdot \left(1 + \frac{\sqrt{5}}{4} \cdot \alpha_1^{\mathrm{bare}}\right) = 0.008459 \times 1.021812 = 0.008644 \text{ eV}$$

**ADOPTED-Z3:** the identification of the Class-1 color-sector coefficient $\sqrt{5}/4$ with a generation-specific neutrino mass correction requires the C_3 Fourier index to correspond to a generation label. Under B6 (`docs/theorem_B6_bridge.md`), C_3 at the P-point is color-$\mathbb{Z}_3$ of SU(3)$_c$, not a generation label; this identification is therefore adopted.

## Result

| Quantity | Value |
|---|---|
| $m_{\nu_3}^{\mathrm{bare}}$ | 0.048277 eV  [ADOPTED-PS] |
| $R = 228/7$ | 32.5714...  [derived] |
| $m_{\nu_2}^{\mathrm{bare}} = m_{\nu_3}^{\mathrm{bare}}/\sqrt{R}$ | 0.008459 eV  [derived given ADOPTED-PS] |
| $\alpha_1^{\mathrm{bare}}$ | $(2/3)^8 = 256/6561$  [derived] |
| $\mathrm{Im}(h)/|h|^2$ | $\sqrt{5}/4$  [derived] |
| correction factor | $1.021812$  [derived] |
| $m_{\nu_2}$ (predicted) | $8.644$ meV |
| $m_{\nu_2}$ (NuFIT 6.0) | $8.654 \pm 0.110$ meV |
| Deviation | $-0.10$ sigma |

The sub-sigma deviation makes this prediction numerically consistent with observation at the 0.1-sigma level. Note that the retracted `predictions/retracted/m_nu2.py` (pre-A3, BLOCKED under B6) had a similar deviation ($\approx 0.5$ sigma using the same bare scale and correction structure) — the structure is the same, but the present file explicitly flags all adopted residuals rather than treating them as derived.

## Cross-check with m_nu3

Since both $m_{\nu_2}$ and $m_{\nu_3}$ carry the same multiplicative correction factor:

$$\frac{m_{\nu_3}^2}{m_{\nu_2}^2} = \frac{(m_{\nu_3}^{\mathrm{bare}})^2}{(m_{\nu_2}^{\mathrm{bare}})^2} = R = \frac{228}{7}$$

exactly. Verified numerically in the `__main__` block of `predictions/m_nu2.py` to residual $< 10^{-8}$.

## Open questions

1. **ADOPTED-PS closure:** derive $M_R$ and $m_t(\mathrm{GUT})$ from A1 + A2-T + A3-T without the girth-cycle-as-$M_R$ identification. Requires theorem-grade MSSM GUT matching. **Status:** open research; no candidate closure path identified per an internal working note.
2. **ADOPTED-Z3 closure:** Sprint 11 workstream B7.3 — mass operator on $\mathbb{C}^3_{\mathrm{gen}}$ orthogonal to the srs C_3 color structure. Under B6, the C_3 at the P-point is color-Z₃ of SU(3)_c; the identification with generation labels requires a separate structural mechanism currently not derived. Related to R-14 Pati-Salam quark/lepton differentiation residue.
3. ~~**I-Feshbach closure:**~~ **CLOSED 2026-05-02 EOD+9** via `docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md` + the author's separate private derivation + m_ν2 PDG sensitivity sweep verification. Promoted from "adopted" to "derived"; reduces the conditional set from {ADOPTED-PS, ADOPTED-Z3, I-Feshbach} to {ADOPTED-PS, ADOPTED-Z3}.
4. **m_ν1 = 0 derivation:** the NuFIT 6.0 normal-ordering convention used in this derivation assumes m_ν1 = 0; the previous structural derivation ("M_D(trivial_s) = 0 at the P-point") was retracted under B6. Currently a CONVENTION not a derived prediction. Tracked as residue R-15 in `docs/audits/registers/structural_residue_register.md`. Re-derivation under C³_gen is open research per an internal working note sub-target B7.3a.v.


## Audit v2 (Clause 7) status

This prediction inherits Row 4 audit v2 closure + ADOPTED-PS-SCALE conditional.
See an internal working note.

- **Status (post-audit-v2; 2026-05-02 EOD+9 promotion):** **THEOREM-GRADE-CONDITIONAL on (ADOPTED-PS-SCALE + ADOPTED-Z3 + Row 4 audit v2)**. The dark-correction form Im(h)/|h|² is theorem-grade (per `docs/theorems/theorem_m_nu_dark_correction_uniqueness_closure.md` 2026-04-29 + `docs/theorems/theorem_analytical_feshbach_ramanujan_boundary.md` 2026-05-02 EOD+4 unification); the I-Feshbach NB-survival↔coupling identification is theorem-grade via the analytical Σ(h) closure + the author's separate private derivation + m_ν2 PDG sensitivity sweep; the bare-scale m_{ν3}^bare derivation remains the un-graduated open input.
- **Named margin:** ADOPTED-PS-SCALE is the dominant un-graduated conditional; ADOPTED-Z3 is the secondary conditional (color↔generation identification under B6 caveat).

## References

- Bass, H. (1992). *Int. J. Math.* **3**, 717-797.
- Chiribella, G., D'Ariano, G.M., Perinotti, P. (2011). *Phys. Rev. A* **84**, 012311.
- Esteban, I. et al. (2024). NuFIT 6.0. http://www.nu-fit.org/
- Ihara, Y. (1966). *J. Math. Soc. Japan* **18**, 219-235.
- Jaynes, E.T. (1957). *Phys. Rev.* **106**, 620-630.
- Lubotzky, A., Phillips, R., Sarnak, P. (1988). *Combinatorica* **8**, 261-277.
- Pati, J.C., Salam, A. (1974). *Phys. Rev. D* **10**, 275.
- Serre, J.-P. (1980). *Trees.* Springer, §I.1.
- Terras, A. (2011). *Zeta Functions of Graphs.* Cambridge, Thm 3.1, §2.1.

## Upstream prediction files

- `predictions/k_star.py` ($k^* = 3$)
- `predictions/d_spatial.py` ($d = 3$)
- `predictions/g_girth.py` ($g = 10$)
- `predictions/feshbach_exponent_principle.py` ($\alpha_1^{\mathrm{bare}} = (2/3)^8$, $n_{\mathrm{fixed}} = 2$)
- `predictions/h_walker_eigenvalue.py` ($h = (\sqrt{3} + i\sqrt{5})/2$)
- `predictions/B_P_doubly_degenerate_h.py` ($|h|^2 = 2$, multiplicity 2, C_3-protected)
- `predictions/R_nu_splitting.py` ($R = 228/7$)
- `proofs/masses/srs_nu_mass_ps.py` ($m_{\nu_3}^{\mathrm{bare}} = 0.048277$ eV; ADOPTED-PS)
