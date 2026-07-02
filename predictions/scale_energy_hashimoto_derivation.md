# Scale energy for dim-6 Lorentz violation effects (E_scale ~ 147 PeV)

## Abstract

The characteristic scale energy at which dim-6 Lorentz violation effects modify photon propagation, derived from the framework's exact η_lattice = 1/12 and the standard pair-production threshold formula E_th = (m_e² E_Pl² / |η|)^(1/4), is approximately 147 PeV. This is ~two orders of magnitude above current LHAASO photon observations (~1 PeV) and within reach of next-generation facilities in the 2030s.

**Result:** E_scale ≈ 1.47 × 10⁸ GeV = 147 PeV.
**Grade:** COMPUTED (Type 2 arithmetic on η_lattice and PDG constants m_e, E_Pl).

## Framework axioms invoked

Via `predictions/eta_lattice_lorentz_dim6.py`:
- **A1**: toggle alphabet.
- **A2 refined**: MDL observer.

## Derivation

### Step 1 — Threshold formula (Type 3, standard physics)

For a dim-6 subluminal LIV dispersion relation of the form

$$E^2 = p^2 + m^2 - \eta \frac{p^4}{E_P^2},$$

the threshold at which LIV effects modify pair-production kinematics is derived via:

$$E_{\text{th}} = \left(\frac{m_e^2 E_P^2}{|\eta|}\right)^{1/4}$$

This is the standard result from Coleman-Glashow 1999 (Phys. Rev. D 59, 116008) and Jacobson-Liberati-Mattingly 2003 (Phys. Rev. D 67, 124011) for dim-6 LIV.

### Step 2 — Framework input η_lattice = 1/12 (Type 4)

From `predictions/eta_lattice_lorentz_dim6.py`: |η_lattice| = 1/12 (CAS-verified at 24+ digit precision).

### Step 3 — Physical constants (external inputs)

| Symbol | Value | Source |
|---|---|---|
| m_e | 0.5109989461 × 10⁻³ GeV | PDG 2024 electron mass |
| E_Pl | 1.2208996 × 10¹⁹ GeV | PDG 2024 Planck energy |

These are genuinely [external] inputs — not framework-derived.

### Step 4 — Numerical evaluation (Type 2)

$$E_{\text{scale}} = \left(\frac{(0.511 \times 10^{-3})^2 \cdot (1.22 \times 10^{19})^2}{1/12}\right)^{1/4} \approx 1.47 \times 10^8 \text{ GeV} = 147 \text{ PeV}.$$

### Result

$$\boxed{E_{\text{scale}} \approx 147 \text{ PeV}.}$$

## Comparison with experiment

Current observational capabilities at UHE photon energies:

| Instrument | Max observed photon energy | Year |
|---|---|---|
| LHAASO KM2A | 1.4 PeV (Cygnus) | 2021 (Nature 594, 33) |
| LHAASO KM2A | 1.1 PeV (Crab) | 2021 (Science 373, 425) |
| HAWC | multi-100 TeV sources | ~2020 |
| IceCube | PeV neutrinos (not photons) | ongoing |

**E_scale ~ 147 PeV is ~two orders of magnitude above current maximum photon observations** (~1.4 PeV). Not currently probed, but:

- **SWGO (Southern Wide-field Gamma-ray Observatory)** — targeting construction in 2020s, sensitivity to ~100 PeV photons;
- **LHAASO upgrades** — planned for 2020s-2030s.

## Open questions

- **Sensitivity to η_lattice sign.** The subluminal (η > 0) case raises pair-production thresholds; observable via anomalous transparency of the universe to UHE photons. See `predictions/universe_transparency.py`.
- **Translation from Hashimoto dispersion coefficient to PHYSICAL photon dispersion.** Step 1's formula assumes a specific relation between the framework's η_lattice (Hashimoto dispersion coefficient at the Ramanujan spectrum) and the physical dim-6 LIV coefficient in the photon propagator. Stage 3 §6.4 states this translation schematically; a full propagator-level derivation is pending.

## References

### Framework
- `predictions/eta_lattice_lorentz_dim6.py` + derivation (framework input).
- `docs/theorems/theorem_lorentz_causal_sector.md` §6.4 (Stage 3 context).

### Published experimental
- **Cao et al. (LHAASO Collab.)** (2021). Nature 594, 33. 1.4 PeV photon observation.
- **Cao et al. (LHAASO Collab.)** (2021). Science 373, 425. Crab PeV photons.
- **Abeysekara et al. (HAWC Collab.)** (2020). Multi-100 TeV sources.

### Published theoretical
- **Coleman, S.R., Glashow, S.L.** (1999). *High-energy tests of Lorentz invariance.* Phys. Rev. D 59, 116008. Original threshold formula for LIV in pair production.
- **Jacobson, T., Liberati, S., Mattingly, D.** (2003). *Threshold effects and Planck scale Lorentz violation.* Phys. Rev. D 67, 124011. Modern LIV threshold analysis.
- **Addazi et al.** (2022). Prog. Part. Nucl. Phys. 125, 103948. Review.
