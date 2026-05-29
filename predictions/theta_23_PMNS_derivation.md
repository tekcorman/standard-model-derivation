# Derivation: θ₂₃ (PMNS Atmospheric Mixing Angle)

**File:** `predictions/theta_23_PMNS.py`  
**Status:** STRICT-SOLID THEOREM-GRADE (Row P13 of `docs/parameters/parameter_uniqueness_ledger.md`, graduated 2026-04-28 via `docs/theorems/theorem_dark_map_class2_closure.md` Theorem 5.1; dark-map Class-2 taxonomy closed). Conditional on Row 18 (C³_obs), Rows 16/17 (PS), A5(b) Level 3 prescription. Clause 8 PASS at −0.37σ from PDG 2024.  
**Date:** 2026-04-19 (initial); status banner refreshed 2026-05-08.  
**Update history:**
- 2026-04-19 session 2: Adopted-I-Feshbach identification subsumed by A5(b) (the coupling clause of A5; `docs/framework/framework_axioms.md` §5b). All references below to "I-Feshbach (adopted)" should be read as "A5(b) (axiom)."
- 2026-04-28: dark-map Class 2 taxonomy gap CLOSED via `docs/theorems/theorem_dark_map_class2_closure.md` Theorem 5.1. Status graduates ADVANCED → STRICT-SOLID THEOREM-GRADE.
- 2026-05-08: status banner refreshed for parameter_linter walk-down session 3.

---

## Abstract

We derive the PMNS atmospheric mixing angle θ₂₃ from the srs lattice geometry without free parameters. The TBM (tri-bimaximal mixing) baseline θ₂₃ = 45° follows from exact C₃ symmetry at the P-point of the srs Brillouin zone. The dark sector breaks this symmetry and shifts θ₂₃ upward by splitting the degenerate ω/ω² eigenvalues. The σ_z=0 theorem — a rigorous consequence of the complex-conjugate structure of the generation eigenstates combined with the reality of the graph perturbation — forces the splitting to be exactly symmetric, yielding

$$\theta_{23} = \arctan\!\left(\frac{1 + \alpha_1^{\rm full}}{1 - \alpha_1^{\rm full}}\right) \approx 48.72°$$

where $\alpha_1^{\rm full} = \tan^2(\arg h) \times \alpha_1^{\rm bare} = \frac{5}{3}\left(\frac{2}{3}\right)^8 = \frac{1280}{19683}$. The predicted value is −0.37σ from the PDG 2024 observed value of 49.2° ± 1.3°. The non-trivial content is the σ_z=0 theorem: most perturbations of a degenerate eigenvalue pair do *not* produce symmetric splitting, but on the srs lattice at P the conjugate-pair structure of the generation eigenstates forces it.

---

## Framework Axioms Invoked

- **A1** (binary self-inverse toggle): determines the srs crystal net as the MDL-selected substrate.
- **A2** (MDL canonicalization): selects k* = 3, g = 10, and the P-point as the MDL saddle.
- **A3** (purification): legitimizes the partial-trace dark sector interpretation; dark sector = trace-out of the purifying auxiliary.
- **A4** (node grading / CAR): establishes fermionic Fock structure at each node.
- **A5** (physical identification): Ramanujan eigenvalues at P = SM spectrum; mass eigenstates correspond to C₃ irrep labels at P.

Additionally:
- **W4** (walker dynamics, Jaynes uniform): per-step NB survival = (k−1)/k, used for α₁_bare.
- **I-Feshbach** (adopted structural theorem): α₁_bare = physical dark-sector coupling magnitude.
- **dark-map Class 2** (adopted classification): θ₂₃ belongs to Class 2 (mass²-class, diagonal C₃), giving dark coefficient tan²(arg h) = 5/3.

---

## Derivation

### Step 1: TBM Baseline θ₂₃ = 45°

At the P-point $\mathbf{k} = (\tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{4})$ of the srs Brillouin zone, the 4×4 Bloch Hamiltonian commutes with the body-diagonal C₃ rotation. The four energy bands decompose under C₃ as:

$$\text{spectrum at P} = 2 \times \text{trivial} \oplus \omega \oplus \omega^2$$

where $\omega = e^{2\pi i/3}$. The ω and ω² bands are degenerate with energy $\lambda_0 = \sqrt{3}$ (Ramanujan saturation: $|h|^2 = k^* - 1 = 2$, so $|h| = \sqrt{2}$ and $E_P = \sqrt{k^*} = \sqrt{3}$).

In the TBM limit (exact C₃ symmetry, no dark correction), the neutrino mass matrix is diagonalized by a unitary with $|U_{\mu 3}| = |U_{\tau 3}|$ (μ–τ symmetry from equal eigenvalue magnitudes), giving:

$$\theta_{23}^{\rm TBM} = \arctan(1) = 45°.$$

*Source:* `predictions/B_P_doubly_degenerate_h.py` (P-point band structure).

### Step 2: σ_z = 0 Theorem

The dark sector couples to the visible sector through a perturbation δH on the graph. Since the graph adjacency matrix is real, δH is real symmetric. The generation eigenstates at P satisfy the conjugate-pair relation:

$$|\omega^2\rangle = e^{i\varphi} \overline{|\omega\rangle}$$

for some global phase $\varphi$. For any real symmetric matrix δH:

$$\langle \omega | \delta H | \omega \rangle
= \sum_{a,b} \psi_{\omega^2}[a]\, \delta H[a,b]\, \psi_\omega[b]$$

(using $\overline{\psi_\omega} = e^{-i\varphi} \psi_{\omega^2}$). Meanwhile:

$$\langle \omega^2 | \delta H | \omega^2 \rangle
= \sum_{a,b} \psi_\omega[a]\, \delta H[a,b]\, \psi_{\omega^2}[b]
= \sum_{a,b} \psi_\omega[b]\, \delta H[b,a]\, \psi_{\omega^2}[a]$$

where the last step uses $\delta H_{ab} = \delta H_{ba}$ (symmetry) with a relabelling $a \leftrightarrow b$. Since diagonal elements of a Hermitian operator are real, and this expression is the complex conjugate of $\langle \omega | \delta H | \omega \rangle$:

$$\langle \omega^2 | \delta H | \omega^2 \rangle = \langle \omega | \delta H | \omega \rangle. \qquad \text{(QED)}$$

The σ_z component in the generation subspace is:

$$(\delta H_{\rm gen})_{zz} = \tfrac{1}{2}\bigl(\langle \omega|\delta H|\omega\rangle - \langle \omega^2|\delta H|\omega^2\rangle\bigr) = 0.$$

Therefore the perturbation decomposes as $\delta H_{\rm gen} = d \cdot I + f \sigma_x + g \sigma_y$ (σ_z absent). The eigenvalue splitting is $\pm\sqrt{f^2+g^2} = \pm|\langle \omega|\delta H|\omega^2\rangle|$, which is **exactly symmetric** around the unperturbed value. Whether the coupling is $\sigma_x$, $\sigma_y$, or a mix is basis-dependent and does not affect the eigenvalue splitting.

*Source:* `proofs/flavor/srs_theta23_sigma_x.py`, Parts D, F, H; Monte Carlo verification (10,000 trials).

### Step 3: Dark Coupling Magnitude

By the Feshbach Exponent Principle (n_fixed = 2, scattering amplitude): $\alpha_1^{\rm bare} = ((k^*-1)/k^*)^{g-2} = (2/3)^8 = 256/6561$.

*Source:* `predictions/feshbach_exponent_principle.py` (combinatorial theorem).

The dark correction to a Class 2 (mass²-class, diagonal C₃) observable carries the coefficient $\tan^2(\arg h)$, which is exact algebra from $h = (\sqrt{3}+i\sqrt{5})/2$:

$$\tan^2(\arg h) = \frac{{\rm Im}(h)^2}{{\rm Re}(h)^2} = \frac{(\sqrt{5}/2)^2}{(\sqrt{3}/2)^2} = \frac{5/4}{3/4} = \frac{5}{3}.$$

*Source:* `predictions/dark_extraction_map.py`, function `dark_coefficient_mass_squared`.

Therefore:

$$\alpha_1^{\rm full} = \frac{5}{3} \times \left(\frac{2}{3}\right)^8 = \frac{1280}{19683}.$$

**Adopted identification 1 (I-Feshbach):** The NB walk survival $\alpha_1^{\rm bare}$ equals the physical dark coupling strength in the Feshbach self-energy. This is an adopted structural theorem; the closure requires explicit computation of the K₄-quotient Feshbach matrix elements (see `../predictions/Feshbach_coupling_strength_derivation.md §9`).

**Adopted identification 2 (dark-map Class 2):** The observable θ₂₃ is a mixing angle from mass-matrix diagonalization, hence diagonal under C₃, hence Class 2. This classification is adopted from `predictions/dark_extraction_map.py`.

### Step 4: Mixing Angle Formula

By the σ_z=0 theorem, the perturbed eigenvalues are:

$$\lambda_\mu = \lambda_0(1 + \alpha_1^{\rm full}), \qquad \lambda_\tau = \lambda_0(1 - \alpha_1^{\rm full}).$$

The atmospheric mixing angle is:

$$\theta_{23} = \arctan\!\left(\frac{\lambda_\mu}{\lambda_\tau}\right)
= \arctan\!\left(\frac{1 + \alpha_1^{\rm full}}{1 - \alpha_1^{\rm full}}\right).$$

---

## Result

$$\boxed{\theta_{23} = \arctan\!\left(\frac{20963}{18403}\right) \approx 48.72°}$$

Exact rational form: $\alpha_1^{\rm full} = 1280/19683$; numerator $= 19683 + 1280 = 20963$; denominator $= 19683 - 1280 = 18403$.

---

## Comparison with Experiment

| Quantity | Value |
|----------|-------|
| Predicted θ₂₃ | 48.72° |
| Observed (PDG 2024) | 49.2° ± 1.3° |
| Deviation | −0.48° (−0.37σ) |
| Observed (NuFIT 6.0, Sep 2024, NO) | 49.0° ± 1.2° |
| Deviation vs NuFIT 6.0 | −0.28° (−0.23σ) |

The prediction is non-maximal mixing (48.72° ≠ 45°). DUNE will measure θ₂₃ to < 0.5° precision, directly testing this prediction.

---

## Open Questions

1. **I-Feshbach closure.** The identification of $\alpha_1^{\rm bare}$ with the physical dark coupling strength requires explicit computation of the Feshbach self-energy matrix elements on the K₄ quotient of the srs lattice. This 12×12 matrix calculation is finite but has not been completed at journal grade. Closure route: `../predictions/Feshbach_coupling_strength_derivation.md §9`.

2. **dark-map Class 2 formal derivation.** The classification of θ₂₃ as Class 2 (mass²-class, diagonal C₃) is adopted from `predictions/dark_extraction_map.py` where it is stated without derivation from A1–A5. A formal proof that mixing angles from mass-matrix diagonalization couple to the dark sector through $\tan^2(\arg h)$ would close this.

3. **μ–τ symmetry origin.** The TBM baseline assumes exact μ–τ symmetry (|E(ω)| = |E(ω²)| at P). This follows from C₃ symmetry at P as a consequence of the srs geometry. The formal proof is in `predictions/B_P_doubly_degenerate_h.py`. No additional input is required beyond what is already closed.

4. **σ_z=0 theorem scope.** The theorem assumes the dark perturbation δH is a real symmetric matrix on the atom-index basis. This holds as long as the dark sector couples through real adjacency weights (unweighted graph). Complex weights (e.g., from spin-orbit coupling) would not satisfy the theorem. Under A1–A5 the graph is real-weighted, so the assumption holds.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
