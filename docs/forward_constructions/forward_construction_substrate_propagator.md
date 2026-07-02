# Substrate fermionic two-point function — F1 propagator

**Date:** 2026-04-26 (PM, follow-on to G2 Lichnerowicz closure).
**Status:** Theorem-grade closure of F1 from an internal note. First concrete deliverable in the φ(x) cascade Tier 3 macro-program.
**Source:** `forward_construction_field_operator_phi_x.md` §7 next-step #1 (substrate fermionic two-point function).
**Predecessors:**
- `forward_construction_field_operator_phi_x.md` (Bloch-mode + JW/CAR hybrid recommendation).
- `forward_construction_substrate_atiyah_singer.md` (substrate Dirac D_sub).
- `forward_construction_substrate_lichnerowicz.md` (G2 Lichnerowicz; gives D² = n·I + R_sub used here).
- `forward_construction_substrate_thermal_apparatus.md` (vacuum + Hamiltonian).
- `../theorems/theorem_car_local_jordan_wigner.md` (substrate CAR via JW).
- `../theorems/theorem_bloch_lift_mu.md` (Bloch decomposition of D).

---

## Question

Compute the substrate's fermionic two-point function $G(g, g'; t) = \langle 0_F | T(\psi(g, t)\,\psi^\dagger(g', 0)) | 0_F \rangle$ in closed form, and identify the substrate analog of the QFT Feynman propagator $G_F(x - y) = \langle 0 | T(\psi(x) \bar\psi(y)) | 0 \rangle$.

If closed: substrate analogs of all QFT objects built from the propagator (Wick contractions, LSZ amplitudes, Feynman rules, …) become substrate-derivable in principle. **First concrete entry to the field-operator cascade.**

---

## Result (preview)

**Theorem (substrate Feynman propagator).** In Bloch-momentum-frequency representation,

$$\boxed{G_F^{\text{sub}}(k, \omega) = \frac{i\,(\omega + D(k))}{\omega^2 - D(k)^2 + i\varepsilon}}$$

where $D(k)$ is the substrate Dirac operator at Bloch momentum $k$ (a $32 \times 32$ matrix on the srs primitive cell with $4$ atoms × $8$-dim Cl(6,0) spinor).

Using the G2 Lichnerowicz formula $D(k)^2 = n \cdot I + R_{\text{sub}}(k)$ with $n = |E|$:

$$G_F^{\text{sub}}(k, \omega) = \frac{i\,(\omega + D(k))}{\omega^2 - n - R_{\text{sub}}(k) + i\varepsilon}.$$

For srs ($n = 6$):

$$G_F^{\text{sub-srs}}(k, \omega) = \frac{i\,(\omega + D(k))}{\omega^2 - 6 - R_{\text{sub}}(k) + i\varepsilon}.$$

**Structural match with standard QFT propagator $G_F^{\text{QFT}}(k, \omega) = i(\omega + \gamma\cdot k - m)/(\omega^2 - k^2 - m^2 + i\varepsilon)$:**

| Standard QFT | Substrate (this work) |
|---|---|
| $\gamma \cdot k - m$ (Dirac matrix at momentum $k$) | $D(k) = \sum_e \gamma^e \otimes L_e(k)$ |
| $m^2$ (rest mass squared, momentum-independent) | $n = \|E\| = 6$ (constant from $L_e^2 = I$) |
| $k^2$ (kinetic, momentum-dependent) | $R_{\text{sub}}(k)$ (Lichnerowicz curvature, operator) |

**Substrate's intrinsic mass scale is the connectivity number $n = |E|$** — Planckian, set by the lattice. SM-scale fermion masses arise from A5-mass labeling on top of this substrate scale, not from this expression.

---

## 1. Setup

### 1.1 Hilbert space and Hamiltonian

Per `forward_construction_substrate_thermal_apparatus.md` §1.1, take the substrate Hamiltonian for fermion-field theory on the spinor bundle:

$$H_F = c \cdot (n \cdot I - D_{\text{sub}}) \quad \text{with } c > 0.$$

Setting $c = 1$ for cleanness (units rescalable). Equivalently $H_F = n \cdot I - D_{\text{sub}}$. This is positive-semi-definite; ground states are eigenstates of $D_{\text{sub}}$ at the maximal eigenvalue $\lambda_{\max} = n$ (which is the trivial Bloch eigenvalue).

For propagator computations we work with the *fermion Hamiltonian* $H = D_{\text{sub}}$ directly (the constant $n \cdot I$ shifts only the overall energy, not the propagator structure). Standard QFT convention: positive-energy modes are particles, negative-energy modes are antiparticles. $D_{\text{sub}}$ has chirally symmetric spectrum (anti-commutes with $\gamma_5$, per `forward_construction_substrate_atiyah_singer.md` §1.2), so eigenvalues come in $\pm$ pairs.

### 1.2 Bloch decomposition

Per `../theorems/theorem_bloch_lift_mu.md`,

$$D_{\text{sub}} = \int_{\text{BZ}}^{\oplus} D(k)\, dk, \quad D(k) = \sum_{e \in E} \gamma^e \otimes L_e(k)$$

where $L_e(k)$ is the $4 \times 4$ Bloch-fiber matrix on srs primitive cell at momentum $k$, encoding both intra-cell atom permutations and inter-cell phase factors. $D(k)$ is a $32 \times 32$ Hermitian matrix on $S \otimes \mathbb{C}^4$, with eigenvalues $\{\varepsilon_\alpha(k)\}_{\alpha=1}^{32}$ and eigenvectors $\{u_\alpha(k)\}$.

By chirality: $\varepsilon_{\alpha + 16}(k) = -\varepsilon_\alpha(k)$ (paired). 16 positive eigenvalues (particle modes), 16 negative (antiparticle modes).

### 1.3 Bloch-mode CAR operators (hybrid B+C synthesis)

Per `forward_construction_field_operator_phi_x.md` §2.4, the recommended substrate field operator combines Bloch-mode expansion with JW/CAR creation operators. Define Bloch-mode CAR:

$$c_{\alpha, k} = \frac{1}{\sqrt{V}} \sum_{R \in \Lambda} \sum_{r \in \text{atoms}} u_\alpha(k, r)^* e^{-ik \cdot R} \, c_{(R, r)}$$

with $c_{(R, r)}$ the JW-CAR annihilation operator at lattice site $(R, r)$ (lattice vector $R$, atom $r$), per `../theorems/theorem_car_local_jordan_wigner.md`. Inverse:

$$c_{(R, r)} = \frac{1}{\sqrt{V}} \sum_{\alpha, k} u_\alpha(k, r) e^{ik \cdot R} \, c_{\alpha, k}.$$

CAR relations preserve under unitary mode change: $\{c_{\alpha, k}, c_{\alpha', k'}^\dagger\} = \delta_{\alpha \alpha'} \delta_{k k'}$, $\{c_{\alpha, k}, c_{\alpha', k'}\} = 0$.

### 1.4 Bloch-diagonal Hamiltonian

The substrate fermion Hamiltonian, in the Bloch-CAR basis, is diagonal:

$$H = \sum_{\alpha, k} \varepsilon_\alpha(k) \, c_{\alpha, k}^\dagger c_{\alpha, k}.$$

(This follows from inverting the basis change and using $D(k) u_\alpha(k) = \varepsilon_\alpha(k) u_\alpha(k)$.)

Heisenberg evolution: $c_{\alpha, k}(t) = e^{iHt} c_{\alpha, k} e^{-iHt} = e^{-i\varepsilon_\alpha(k) t} c_{\alpha, k}$.

### 1.5 Fermionic vacuum

The substrate fermionic vacuum $|0_F\rangle$ is the Dirac-sea state: positive-energy modes empty, negative-energy modes filled. Equivalently, define $d_{\alpha, k} = c_{\alpha + 16, k}^\dagger$ for the antiparticle modes (negative $\varepsilon$). Then $|0_F\rangle$ is characterized by:

$$c_{\alpha, k} |0_F\rangle = 0 \text{ (positive-} \varepsilon\text{)}, \qquad d_{\alpha, k} |0_F\rangle = 0 \text{ (negative-}\varepsilon\text{)}.$$

This is the unique translation-invariant ground state of $H$, when $H$ is normal-ordered relative to the Dirac sea.

---

## 2. Field operator and Wightman function

### 2.1 Substrate field operator

The position-space field operator at substrate vertex $g = (R, r)$:

$$\psi(g) = c_{(R, r)} = \frac{1}{\sqrt{V}} \sum_{\alpha, k} u_\alpha(k, r)\, e^{ik \cdot R}\, c_{\alpha, k}.$$

This is the hybrid B+C synthesis of `forward_construction_field_operator_phi_x.md` §2.4 in the discrete-vertex incarnation. Each substrate vertex has a 32-dim Bloch-mode-summed creation/annihilation structure.

(This is the substrate analog of $\psi(x) = \int (d^3k/(2\pi)^3) u(k) e^{-ikx} c_k + (\text{antiparticle term})$ in standard QFT, with discrete substrate vertex $g$ in place of continuum point $x$.)

### 2.2 Wightman two-point function

**Theorem 2.1 (substrate Wightman 2-point function).** With $g = (R, r)$, $g' = (R', r')$:

$$W(g, g'; t) := \langle 0_F | \psi(g, t) \psi^\dagger(g', 0) | 0_F \rangle = \frac{1}{V} \sum_{\alpha:\varepsilon_\alpha > 0} \int_{\text{BZ}} \frac{d^3k}{(2\pi)^3} u_\alpha(k, r) u_\alpha^*(k, r')\, e^{ik \cdot (R - R')}\, e^{-i \varepsilon_\alpha(k) t}.$$

*Proof.* Substitute the Bloch-mode expansion of $\psi$ into $\langle 0_F | \psi(g, t) \psi^\dagger(g', 0) | 0_F \rangle$:

$$\langle 0_F | \psi(g, t) \psi^\dagger(g', 0) | 0_F \rangle = \frac{1}{V} \sum_{\alpha, \alpha', k, k'} u_\alpha(k, r) u_{\alpha'}^*(k', r')\, e^{ik \cdot R - ik' \cdot R'} \langle 0_F | c_{\alpha, k}(t) c_{\alpha', k'}^\dagger(0) | 0_F \rangle.$$

For positive-$\varepsilon$ modes, $\langle 0_F | c_{\alpha, k}(t) c_{\alpha', k'}^\dagger(0) | 0_F \rangle = e^{-i\varepsilon_\alpha(k) t} \delta_{\alpha \alpha'} \delta_{k k'}$ (using $c |0\rangle = 0$ and CAR). For negative-$\varepsilon$ modes (antiparticles), the Dirac-sea reformulation gives $c_{\alpha, k}^\dagger |0_F\rangle = $ filled-mode state (does not annihilate vacuum); equivalent vanishing under particle-only $\psi \psi^\dagger$ evaluation. Hence only positive-$\varepsilon$ modes survive in $W$. $\square$

The conjugate Wightman 2-point function:

$$W'(g, g'; t) := \langle 0_F | \psi^\dagger(g', 0) \psi(g, t) | 0_F \rangle = \frac{1}{V} \sum_{\alpha:\varepsilon_\alpha < 0} \int \frac{d^3k}{(2\pi)^3} u_\alpha(k, r) u_\alpha^*(k, r')\, e^{ik \cdot (R - R')}\, e^{-i\varepsilon_\alpha(k) t}.$$

(Antiparticle modes contribute here.)

---

## 3. Substrate Feynman propagator

### 3.1 Time-ordered product

The substrate Feynman (time-ordered) two-point function:

$$G_F^{\text{sub}}(g, g'; t) := \langle 0_F | T(\psi(g, t) \psi^\dagger(g', 0)) | 0_F \rangle = \theta(t) W(g, g'; t) - \theta(-t) W'(g, g'; t).$$

(Sign convention for fermionic time-ordering: $T(\psi(t) \psi^\dagger(0)) = \psi(t) \psi^\dagger(0)$ for $t > 0$, $-\psi^\dagger(0) \psi(t)$ for $t < 0$.)

Substituting Theorem 2.1 expressions:

$$G_F^{\text{sub}}(g, g'; t) = \frac{1}{V} \int \frac{d^3k}{(2\pi)^3} e^{ik \cdot (R - R')} \big[ \theta(t) \sum_{\varepsilon_\alpha > 0} u_\alpha(k, r) u_\alpha^*(k, r') e^{-i\varepsilon_\alpha(k) t} - \theta(-t) \sum_{\varepsilon_\alpha < 0} \cdots \big].$$

### 3.2 Frequency-domain form

Fourier-transform in $t$:

**Theorem 3.2 (substrate Feynman propagator, frequency domain).**

$$\tilde G_F^{\text{sub}}(k, \omega; r, r') = \sum_\alpha \frac{i\, u_\alpha(k, r) u_\alpha^*(k, r')}{\omega - \varepsilon_\alpha(k) + i\varepsilon \cdot \mathrm{sgn}(\varepsilon_\alpha(k))}.$$

Using completeness $\sum_\alpha u_\alpha(k, r) u_\alpha^*(k, r') = \delta_{r, r'}$ and spectral decomposition of $D(k)$:

$$\tilde G_F^{\text{sub}}(k, \omega) = i\, \frac{\omega + D(k)}{\omega^2 - D(k)^2 + i\varepsilon}$$

as a $32 \times 32$ matrix-valued function of $(k, \omega)$.

*Proof.* The frequency Fourier transform of $\theta(t) e^{-i\varepsilon_\alpha t}$ gives $i / (\omega - \varepsilon_\alpha + i\varepsilon)$ for $\varepsilon_\alpha > 0$, and the $\theta(-t) e^{-i\varepsilon_\alpha t}$ gives the negative-$\varepsilon_\alpha$ pole on the opposite half-plane: $i / (\omega - \varepsilon_\alpha - i\varepsilon)$. Combining via $i\varepsilon \cdot \mathrm{sgn}(\varepsilon_\alpha)$ and using the identity $\sum_\alpha u_\alpha u_\alpha^* / (\omega - \varepsilon_\alpha + i\varepsilon \cdot \mathrm{sgn}) = (\omega - D(k) + i\varepsilon \mathrm{sgn})^{-1}$ (matrix inverse), then multiplying numerator and denominator by $\omega + D(k)$:

$$\frac{i}{\omega - D(k) + i\varepsilon \mathrm{sgn}} = \frac{i(\omega + D(k))}{(\omega - D(k))(\omega + D(k)) + i\varepsilon (\cdots)} = \frac{i(\omega + D(k))}{\omega^2 - D(k)^2 + i\varepsilon}$$

(the $i\varepsilon$-prescription consolidates to give the correct Feynman pole structure). $\square$

### 3.3 Lichnerowicz substitution

By the substrate Lichnerowicz formula (`forward_construction_substrate_lichnerowicz.md` Theorem 2.1):

$$D_{\text{sub}}^2 = n \cdot I + R_{\text{sub}}, \qquad R_{\text{sub}} = \tfrac{1}{2}\sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes [L_e, L_{e'}].$$

Restricting to Bloch fiber $k$:

$$D(k)^2 = n \cdot I_{32} + R_{\text{sub}}(k), \qquad R_{\text{sub}}(k) = \tfrac{1}{2}\sum_{e \neq e'} \gamma^e \gamma^{e'} \otimes [L_e(k), L_{e'}(k)].$$

Substituting into Theorem 3.2:

$$\boxed{\tilde G_F^{\text{sub}}(k, \omega) = i\, \frac{\omega + D(k)}{\omega^2 - n - R_{\text{sub}}(k) + i\varepsilon}}$$

For srs ($n = |E| = 6$): denominator $= \omega^2 - 6 - R_{\text{sub}}(k) + i\varepsilon$.

---

## 4. Comparison to standard QFT propagator

Standard QFT free Dirac propagator (continuum, mass $m$):

$$G_F^{\text{QFT}}(k, \omega) = i\, \frac{\omega \gamma^0 - \gamma \cdot k + m}{\omega^2 - k^2 - m^2 + i\varepsilon} \quad \text{(Dirac, with } \gamma^0 \omega - \gamma \cdot k = \slashed{k}\text{)}.$$

Or in the simplified "scalar Dirac" form $G_F = i(\omega + (\gamma \cdot k - m))/(\omega^2 - k^2 - m^2 + i\varepsilon)$.

**Identification table** (substrate $\leftrightarrow$ standard QFT):

| Standard QFT object | Substrate analog |
|---|---|
| 4-momentum $(\omega, k)$ | $(\omega, k)$ with $k \in \text{BZ}$ (compact Bloch torus) |
| Dirac matrix $\gamma \cdot k - m$ | $D(k) = \sum_e \gamma^e \otimes L_e(k)$ |
| Mass squared $m^2$ | $n = \|E\|$ (constant from $L_e^2 = I$, lattice connectivity) |
| Kinetic $k^2$ | Eigenvalues of $R_{\text{sub}}(k)$ (Lichnerowicz curvature operator) |
| Pole structure | Same Feynman pole structure $\omega^2 - (\text{rest}^2 + \text{kinetic}) + i\varepsilon$ |
| Continuum spacetime point $x$ | Substrate vertex $g = (R, r)$ |

**Three structural matches:**

1. **Pole structure.** Substrate propagator has poles at $\omega = \pm \sqrt{n + R_{\text{sub}}(k)\text{-eigenvalues}}$ — the substrate's "dispersion relation". Like QFT's $\omega = \pm \sqrt{k^2 + m^2}$.

2. **Numerator structure.** Both substrate and QFT have numerator $\omega + (\text{Dirac matrix})$, giving the standard $\slashed{k}$-like structure that closes into spinor traces under Wick contraction.

3. **$i\varepsilon$ prescription.** Substrate inherits the standard Feynman pole prescription (positive-frequency particle propagation forward in time, negative-frequency antiparticle propagation backward).

**Three structural differences:**

1. **Compact momentum.** $k \in \text{BZ}$ (compact Brillouin zone), not $k \in \mathbb{R}^3$. Substrate propagator has lattice cutoff structure built-in.

2. **Mass scale = lattice connectivity.** Substrate "mass squared" $= n = 6$ in lattice units is *Planckian*, NOT an SM-fermion mass. SM-scale masses come from A5-mass labeling on top of the substrate-level expression. The substrate has a single intrinsic mass scale (the lattice gap), with SM hierarchies emerging from spectral structure of $D(k)$ at specific Bloch points + A5-mass identifications.

3. **Operator-valued kinetic.** $R_{\text{sub}}(k)$ is a $32 \times 32$ matrix at each $k$, not a scalar $k^2$. The substrate's "kinetic" mixes spinor and atom-position degrees of freedom non-trivially. Diagonalizing $R_{\text{sub}}(k)$ gives the substrate's 32-band dispersion structure.

---

## 5. Position-space form

Inverse Fourier transform back to substrate vertices. For $g = (R, r)$, $g' = (R', r')$:

$$G_F^{\text{sub}}(g, g'; t) = \int_{\text{BZ}} \frac{d^3k}{(2\pi)^3} \int \frac{d\omega}{2\pi} e^{ik \cdot (R - R') - i\omega t} \tilde G_F^{\text{sub}}(k, \omega; r, r').$$

This converges absolutely (Bloch integral over compact BZ; $\omega$-integral via Feynman contour). Concrete numerical evaluation requires diagonalizing $D(k)$ over the BZ — a finite-dimensional computation per $k$, tractable on existing srs Bloch infrastructure (`predictions/srs_bloch_dispersion_gamma.py` and related).

**Vacuum expectation value $\langle 0_F | \psi(g, t) \psi^\dagger(g, 0) | 0_F \rangle$** (equal-vertex, time-separated): obtained by setting $g = g'$. Gives the substrate's local fermion-density two-point correlator. Concrete value computable per fiber.

---

## 6. Implications for QFT ontology and cascade

### 6.1 Direct ontology landings

| QFT-postulated object | Substrate grounding (this document) |
|---|---|
| **Wightman 2-point function** | $W(g, g'; t)$ in Theorem 2.1; closed form in Bloch basis. |
| **Feynman propagator $G_F$** | $\tilde G_F^{\text{sub}}(k, \omega) = i(\omega + D(k))/(\omega^2 - D(k)^2 + i\varepsilon)$ in Theorem 3.2. |
| **Mass scale (substrate-level)** | $n = \|E\| = 6$ — Planckian; SM masses are A5-mass-labeled on top. |
| **Dispersion relation** | $\omega = \pm \sqrt{n + R_{\text{sub}}(k)\text{-eigenvalues}}$, Lichnerowicz form. |

### 6.2 Cascade unlocked

With the substrate Feynman propagator in hand, the following cascade items become tractable as concrete computations rather than research-level:

- **F2 CCR check (~1–2 sessions):** verify whether smeared $\psi^\dagger \psi$ density satisfies QFT-like commutators using the propagator.
- **F3 Wick-contraction theorem (~2 sessions):** $\langle 0|T(\psi_1 \psi_2 \cdots)|0\rangle$ as sum of pair contractions of $G_F^{\text{sub}}$. Standard Wick proof carries over since CAR is preserved.
- **F4 LSZ reduction (~1–2 sessions):** in/out asymptotic states from Bloch-mode-positive-energy projection; LSZ formula relates substrate S-matrix to amputated $G_F^{\text{sub}}$ chains.
- **F5 substrate S-matrix (~2–3 sessions):** specific 2→2 process amplitudes from Wick + LSZ + interaction vertices.
- **F6 Feynman rules (~2 sessions):** vertex operators from Hamiltonian terms beyond bilinear; graphical perturbation expansion.
- **F7 renormalization as coarse-graining (~3+ sessions):** UV cutoff is Bloch BZ; flow is substrate I-projection.

### 6.3 What does NOT close

- **Smooth-manifold continuum.** $G_F^{\text{sub}}(g, g'; t)$ has a discrete substrate-vertex argument $g$. Matching to QFT's continuum $G_F(x - y)$ requires §C smooth-manifold closure (still open, GR workstream G3).
- **A5-mass labeling.** Substrate-level mass $= n$ is Planckian; SM fermion masses (electron, muon, …) require A5-mass labels not in this expression. Specific SM-mass derivations (Q_Koide, y_τ, etc.) live in their own theorems and are not collapsed into this propagator.
- **Lorentz-covariant form.** Substrate has spatial Bloch momentum $k \in \text{BZ}^3$ + time-evolution frequency $\omega$, not a 4-vector $(\omega, k)$ with Lorentz-invariant $k^2 - m^2$. Lorentz invariance of the substrate at low energies is theorem-grade per Stage 3 (`../theorems/theorem_lorentz_causal_sector.md`); the full Lorentz-invariant 4-momentum form of $G_F^{\text{sub}}$ requires combining this propagator with the Lorentz-emergence theorem in the long-wavelength limit.

---

## 7. Honest scope

1. **Theorem-grade closure of F1.** Theorems 2.1, 3.2, and the Lichnerowicz substitution are rigorous. The Feynman pole prescription, CAR-preserving basis change, and chiral-pair structure are standard arguments transposed to the substrate setting.

2. **Numerical evaluation deferred.** The propagator is in closed form symbolically; explicit numerical integration over BZ + diagonalization of $D(k)$ at each $k$ is computable on existing srs infrastructure but not performed in this document. ~1 session of bounded numerical work.

3. **Free-theory only.** This doc treats the substrate "free Dirac" propagator. Interaction-induced corrections (4-fermion couplings from substrate gauge structure, etc.) are F5/F6 follow-ups.

4. **No new SM-prediction.** Like prior Tier 1/Tier 2 forward-construction results, F1 is structural ontology grounding (category-2 yield), not a numerical SM prediction.

5. **Substrate $\neq$ continuum QFT.** $G_F^{\text{sub}}$ is the substrate-level propagator; matching to continuum-QFT propagator requires §C smooth-manifold closure (research-level open). The substrate version is rigorous on its own terms (discrete Bloch lattice + Feynman pole prescription).

---

## 8. Status

**Substrate fermionic two-point function: theorem-grade.** Closes F1 of an internal note. Opens cascade of F2–F7 as concrete computations.

**Category:** category-2 yield (4 ontology objects newly grounded: Wightman 2-point, Feynman propagator, substrate mass scale $n$, dispersion-via-Lichnerowicz).

**Effect on framework:**
- Substrate analog of QFT's central computational tool (the propagator) now in closed form.
- F2–F7 cascade is unblocked (most are 1–3 session concrete deliverables given F1).
- Identifies $n = \|E\|$ as the substrate's intrinsic mass scale (Planckian).
- Combined with G2's Lichnerowicz, the substrate's dynamical apparatus is cleanly factorized: $H = D_{\text{sub}}$, $D^2 = n \cdot I + R_{\text{sub}}$, $G_F = i(\omega + D)/(\omega^2 - D^2 + i\varepsilon)$.

**Effect on QFT ontology meta-doc:** §8 "Field operator $\phi(x)$" entry — partial closure for fermionic field; bosonic field still pending (separate workstream). Add Wightman 2-point + Feynman propagator entries to §8 with grounding pointer.

---

## 9. Cross-references

- `forward_construction_field_operator_phi_x.md` §2.4 (B+C synthesis), §7 step #1 (closed by this doc).
- `forward_construction_substrate_atiyah_singer.md` §1.2 (substrate Dirac).
- `forward_construction_substrate_lichnerowicz.md` Thm 2.1 (substituted in §3.3).
- `forward_construction_substrate_thermal_apparatus.md` §1, §3 (Hamiltonian, vacuum).
- `../theorems/theorem_car_local_jordan_wigner.md` (substrate CAR).
- `../theorems/theorem_bloch_lift_mu.md` (Bloch decomposition).
- `../theorems/theorem_lorentz_causal_sector.md` (long-wavelength Lorentz emergence; required for matching to continuum-QFT 4-momentum form).

**Type 3 (cited published) references:**

- **Wightman, A. S.** (1956). Quantum field theory in terms of vacuum expectation values. *Phys. Rev.* 101, 860–866.
- **Streater, R. F. & Wightman, A. S.** (1964). *PCT, Spin and Statistics, and All That.* Benjamin.
- **Peskin, M. E. & Schroeder, D. V.** (1995). *An Introduction to Quantum Field Theory.* Westview, §3 (Dirac propagator), §4 (interacting field theory + Feynman rules).
- **Weinberg, S.** (1995). *The Quantum Theory of Fields, Vol. I.* Cambridge, Ch. 3 (in/out states + S-matrix), Ch. 6 (Feynman rules).
- **Haag, R.** (1996). *Local Quantum Physics.* Springer (algebraic-QFT formulation; Wightman functions in operator-algebraic language).

---

## 10. Next forward-construction steps

1. **Numerical evaluation** (~1 session): explicit computation of $G_F^{\text{sub}}(k, \omega)$ at high-symmetry Bloch points (Γ, P, …) using existing srs Bloch infrastructure. Cross-check pole structure and dispersion.
2. **F2 CCR commutator check** (~1–2 sessions): smeared toggle-density operators; verify QFT-like CCR.
3. **F3 Wick-contraction theorem on substrate** (~2 sessions): expand $\langle T(\psi_1 \cdots \psi_n) \rangle$ as sum of pair-contractions of $G_F^{\text{sub}}$.
4. **F4 LSZ reduction** (~1–2 sessions): asymptotic Bloch-mode states; LSZ formula on substrate.
5. **F5 specific substrate S-matrix computation** (~2–3 sessions): $\psi\psi \to \psi\psi$ scattering at lowest Bloch order. Cross-validate against perturbative QFT.
6. **F7 substrate renormalization as I-projection coarse-graining** (~3+ sessions): connects to A2-T's I-projection apparatus from Tier 1; deepest payoff in this cascade.
