# Universe transparency threshold (E_transparent ~ 147 PeV)

## Abstract

Subluminal dim-6 Lorentz violation (η_lattice = 1/12 > 0) raises the standard QED pair-production threshold E_γ·E_bg > m_e² for UHE photons interacting with the cosmic microwave background and extragalactic background light. Consequently, the universe becomes progressively more transparent to UHE photons above E ~ 147 PeV (framework's scale energy). This is a distinct-from-standard-model observable: current observations show tentative evidence (anomalous 18-TeV photon from GRB 221009A, Finke & Razzaque 2023) qualitatively consistent in sign but not yet probing the 147 PeV scale.

**Result:** E_transparent ≈ 147 PeV (onset of transparency enhancement).
**Grade:** COMPUTED (derives from η_lattice and scale energy; same order-of-magnitude analysis as scale_energy_hashimoto.py).

## Framework axioms invoked

Via `predictions/eta_lattice_lorentz_dim6.py`:
- A1 (toggle alphabet), A2 refined (MDL observer).

## Derivation

### Step 1 — Standard QED threshold (Type 3)

The pair-production process γ + γ_bg → e⁺ e⁻ has threshold (in the absence of LIV):

$$E_\gamma \cdot E_{bg} > m_e^2 c^4.$$

For background photons at CMB (~meV) or EBL (~eV) scales, the corresponding threshold photon energies span the PeV to EeV range.

### Step 2 — Subluminal dim-6 LIV shifts threshold up (Type 3)

With dim-6 LIV in the photon dispersion:

$$E_\gamma^2 = p_\gamma^2 - \eta \frac{p_\gamma^4}{E_P^2},$$

for η > 0 (subluminal), at fixed p_γ the photon's center-of-momentum energy for interaction with a soft-background photon decreases, raising the threshold. Reference: Jacobson-Liberati-Mattingly 2003, Phys. Rev. D 67, 124011.

### Step 3 — Onset scale (Type 2, same as scale_energy_hashimoto)

The crossover scale at which LIV effects become significant for pair production is the same as the scale energy:

$$E_{\text{transparent}} = \left(\frac{m_e^2 E_P^2}{\eta_{\text{lattice}}}\right)^{1/4}.$$

With η_lattice = 1/12 (framework), m_e = 0.511 MeV, E_Pl = 1.22 × 10¹⁹ GeV:

$$E_{\text{transparent}} \approx 147 \text{ PeV.}$$

Above this scale, the universe becomes increasingly transparent to UHE photons.

### Step 4 — Sign criterion (Type 2)

Transparency enhancement requires η > 0 (subluminal). The framework predicts η_lattice = +1/12 > 0 from the Hashimoto dispersion's sign (Stage 3 §6.1), matching the required sign.

### Result

$$\boxed{E_{\text{transparent}} \approx 147 \text{ PeV with subluminal sign.}}$$

Above this energy, UHE photons from cosmological distances are more likely to reach Earth than standard QED predicts.

## Comparison with experiment

| Observation | Relevance |
|---|---|
| GRB 221009A, anomalous 18-TeV photon (Finke & Razzaque, ApJL 942, L21 (2023) [arXiv:2210.11261]) | Tentative evidence for raised pair-production thresholds. Sign consistent with η > 0. Best-fit E_QG,2 ≲ 10⁻⁶ E_Pl, not literally compatible with η_lattice = 1/12 but same direction. |
| Pierre Auger UHE-photon flux upper limits (JCAP 05 (2023) 021) | Upper bounds on photon fluxes above 10¹⁷ eV ≈ 100 PeV. Does not yet constrain transparency at 147 PeV. |
| LHAASO PeV photon observations (Nature 594, 33 (2021); Science 373, 425 (2021)) | Up to 1.4 PeV from galactic sources; not probing transparency regime. |

**Qualitative consistency:** existing evidence points in the right sign direction (raised thresholds for UHE photons) but quantitative tests of the 147 PeV scale require next-generation facilities (SWGO, LHAASO upgrades in the 2030s).

## Open questions

- **Quantitative transparency curve.** The Step 3 formula gives the ONSET scale. A full transparency model (fraction of UHE photons surviving cosmic propagation as a function of energy and redshift) requires integrating the modified pair-production cross section over background photon fields — standard computation with framework η as input, not done here.
- **Translation from Hashimoto dispersion to physical photon propagator.** Same open question as `scale_energy_hashimoto_derivation.md`: the framework provides η from the Hashimoto (NB walk) dispersion; its physical identification as the dim-6 LIV coefficient in the propagating photon is a Stage 3 §6.4 schematic argument.
- **GRB 221009A quantitative fit.** Reconciling the tentative Finke-Razzaque best-fit value with the framework's specific η = 1/12 is pending — could be a source of confirmation or tension as more data accumulate.

## References

### Framework
- `predictions/eta_lattice_lorentz_dim6.py` + derivation.
- `predictions/scale_energy_hashimoto.py` + derivation.
- `docs/theorems/theorem_lorentz_causal_sector.md` §6.4 and §11.

### Published experimental
- **Finke & Razzaque** (2023). ApJL 942, L21 [arXiv:2210.11261]. Tentative LIV evidence from GRB 221009A 18-TeV photon.
- **Pierre Auger Collab.** (2023). JCAP 05, 021. UHE photon flux upper limits above 10¹⁷ eV.
- **Cao et al. (LHAASO)** (2021). Nature 594, 33; Science 373, 425. PeV photons.

### Published theoretical
- **Jacobson, Liberati, Mattingly** (2003). *Threshold effects and Planck scale Lorentz violation.* Phys. Rev. D 67, 124011.
- **Martinez-Huerta et al.** (2020). Symmetry 12, 1232. Review of UHE-photon transparency with LIV.
- **Coleman, S.R., Glashow, S.L.** (1999). Phys. Rev. D 59, 116008.
