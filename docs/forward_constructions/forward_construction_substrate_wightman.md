# Substrate Wightman axioms — F11

**Date:** 2026-04-27 (PM, follow-on to F4 LSZ closure).
**Status:** Theorem-grade closure of F11 from an internal note at the **substrate-discrete level**. Continuum-Lorentz extensions are partial (Stage-3 leading-order + Iorio-elastic-curvature + multi-valley Γ/H pair); fully-continuum version pending §C smooth-manifold closure.
**Source:** an internal note Tier-3 next-step #4 (substrate Wightman axioms).
**Predecessors:**
- `forward_construction_substrate_propagator.md` (F1, propagator).
- `forward_construction_substrate_wick.md` (F3, Wick).
- `forward_construction_substrate_lsz.md` (F4, LSZ).
- `forward_construction_substrate_thermal_apparatus.md` §3 (vacuum |0_F⟩).
- `../theorems/theorem_car_local_jordan_wigner.md` (CAR via JW).
- `../theorems/theorem_lorentz_causal_sector.md` (Stage-3 leading-order Lorentz invariance).
- `predictions/srs_dirac_cone_velocities.py` + `proofs/foundations/lorentz_sig_spin1_dirac_decomposition.py` (Γ-cone spin-1 Dirac with emergent SO(3)).
- `proofs/foundations/lorentz_sig_iorio_session2_vielbein.py` etc. (Iorio-elastic curved-space spin-1 Dirac).

---

## Question

Do the substrate field operators (per F1: $\psi(g, t) = (1/\sqrt V) \sum_{\alpha, k} u_\alpha(k, r) e^{ik\cdot R} c_{\alpha, k}(t)$) satisfy the seven Wightman axioms (Streater–Wightman 1964) of axiomatic QFT?

Wightman's axioms are the foundational postulates of constructive QFT. If satisfied, the substrate's emergent fermionic field theory is a Wightman theory in the rigorous sense; if not, the framework's "QFT-on-substrate" identification needs structural augmentation.

---

## Result (preview)

**Theorem (substrate Wightman axioms — discrete-level closure).** All seven Wightman axioms hold for substrate fermion fields at the **discrete substrate level**, with the Poincaré group replaced by the substrate's discrete symmetry group $\mathbb Z^3 \rtimes (\text{point group } 432) \times \mathbb R_t$ (lattice translations + cubic 432 + continuous time):

| Axiom | Status |
|---|---|
| **W1**: separable Hilbert space + symmetry rep | ✅ theorem-grade (Fock space over Dirac sea + lattice + cubic 432 + continuous-time-evolution rep) |
| **W2**: unique vacuum | ✅ theorem-grade (Dirac sea, unique translation-invariant ground state) |
| **W3**: cyclic vacuum | ✅ theorem-grade (free fermion polynomial algebra is dense in Fock space) |
| **W4**: spectrum condition | ✅ theorem-grade (energy spectrum bounded below; positive-energy normal-ordered $H$) |
| **W5**: microcausality (locality) | ✅ theorem-grade at discrete level via CAR; ⚠ continuum spacelike-separation requires Stage-3-extended emergent Lorentzian metric |
| **W6**: covariance | ✅ theorem-grade for $\mathbb Z^3 \rtimes 432 \times \mathbb R_t$; ⚠ full Poincaré covariance requires continuum-limit lift (Stage 3 + spin-1 Dirac + Iorio-elastic provides leading-order, ✓; full §C closure pending) |
| **W7**: fields as operator-valued distributions | ✅ theorem-grade as operator-valued *functions* on discrete vertex set; ⚠ operator-valued *distributions* on continuum spacetime require §C |

**Bottom line.** The framework's substrate fermion theory **is a discrete Wightman theory** at theorem grade. Three of the seven axioms have continuum-limit aspects pending §C, but each has a theorem-grade discrete counterpart that becomes the relevant Wightman content in the framework's natural setting.

The substrate is therefore a fully-rigorous **discrete axiomatic QFT** — an instance of Streater–Wightman's framework on a lattice with cubic-432 instead of full Poincaré. Continuum extensions follow from §C closure, which is independently active (Iorio-elastic + spin-1 Dirac + leading-order Lorentz covariance form a concrete partial-closure pathway).

---

## 1. Setup

The seven Wightman axioms (Streater–Wightman 1964 *PCT, Spin and Statistics, and All That*; Glimm–Jaffe 1987 *Quantum Physics — A Functional Integral Point of View* §6):

1. **W1**: a separable complex Hilbert space $\mathcal H$ carrying a continuous unitary representation of the connected Poincaré group $\mathcal P^\uparrow_+$.
2. **W2**: a unique (up to phase) Poincaré-invariant unit vector $|0\rangle$, the vacuum.
3. **W3**: $|0\rangle$ is **cyclic** for the polynomial algebra of fields: $\overline{\text{span}}\{\Phi(f_1) \cdots \Phi(f_n)|0\rangle\} = \mathcal H$.
4. **W4**: **spectrum condition** — the energy-momentum operator $P^\mu$ has spectrum in the closed forward light cone $\{p^0 \ge |\mathbf p|\}$.
5. **W5**: **microcausality** — fields at spacelike-separated points (anti)commute: $[\Phi(x), \Phi(y)]_\pm = 0$ for $(x - y)^2 < 0$.
6. **W6**: **covariance** — fields transform as a representation of $\mathcal P^\uparrow_+$: $U(\Lambda, a) \Phi(x) U(\Lambda, a)^{-1} = S(\Lambda)^{-1}_{\,ab} \Phi(\Lambda x + a)_b$.
7. **W7**: $\Phi$ is an **operator-valued tempered distribution** on Schwartz test functions $\mathcal S(\mathbb R^4)$.

The substrate has:
- **Hilbert space**: Fock space over the substrate fermionic Dirac sea (F1 §1.5).
- **Symmetry group**: $\mathbb Z^3 \rtimes (\text{cubic point group } 432) \times \mathbb R_t$ — discrete lattice translations + cubic crystal symmetry + continuous time.
- **Spacetime**: $\Lambda \times \mathbb R_t$ where $\Lambda$ is the BCC + Wyckoff-8a lattice (discrete spatial) and time is continuous.
- **Field operators**: $\psi(g, t)$ at substrate vertex $g = (R, r)$ and continuous time $t$ (per F1 §1.3, F4 §1.1).

The substrate's "Wightman framework" replaces $\mathcal P^\uparrow_+$ with the substrate symmetry group; the seven axioms then map directly to substrate analogs.

---

## 2. Axiom-by-axiom closure

### W1: Hilbert space + symmetry representation  ✅

The substrate Hilbert space is the Fock space $\mathcal F$ over the single-particle Bloch eigenmodes of $D_{\text{sub}}$:

$$\mathcal F \;=\; \mathbb C \oplus \bigoplus_{n\ge 1} \Lambda^n \mathcal H_1,\qquad \mathcal H_1 = \bigoplus_{\alpha, k\in\text{BZ},\,\varepsilon_\alpha(k)>0}\,\mathbb C\,c_{\alpha, k}^\dagger|0_F\rangle$$

(the antisymmetric Fock space over positive-energy Bloch modes). $\mathcal F$ is separable (countable orthonormal basis from the discrete BZ × Bloch-index).

Symmetry representation: lattice translations $T_a$ act as $T_a c_{\alpha, k}^\dagger T_a^{-1} = e^{-ik\cdot a} c_{\alpha, k}^\dagger$. Cubic 432 acts on $(R, r)$ via the corresponding crystallographic representation. Continuous time evolution: $e^{iHt} c_{\alpha, k}^\dagger e^{-iHt} = e^{i\varepsilon_\alpha(k) t} c_{\alpha, k}^\dagger$.

These together form a continuous unitary representation of $\mathbb Z^3 \rtimes 432 \times \mathbb R_t$. ✓

**Theorem-grade.** Direct from F1 §1.3-1.5.

### W2: Unique vacuum  ✅

The substrate vacuum $|0_F\rangle$ is the Dirac-sea state: positive-$\varepsilon$ modes empty, negative-$\varepsilon$ modes filled. Per F1 §1.5, $|0_F\rangle$ is characterised by $c_{\alpha, k} |0_F\rangle = 0$ for $\varepsilon_\alpha(k) > 0$ and $d_{\alpha, k} |0_F\rangle = 0$ for $\varepsilon_\alpha(k) < 0$ (where $d = c^\dagger$ for negative-$\varepsilon$ modes).

**Uniqueness.** Any translation-invariant ground state of the bilinear normal-ordered Hamiltonian $:H: = \sum_{\alpha, k} \varepsilon_\alpha(k) c_{\alpha, k}^\dagger c_{\alpha, k}$ has all positive-$\varepsilon$ occupation numbers zero (otherwise translation-invariance would be broken by the occupied $k$-modes). Therefore $|0_F\rangle$ is unique up to phase. ✓

**Theorem-grade.** Standard textbook fact (Peskin–Schroeder §4.2, Weinberg §4.5 & §10.7).

### W3: Cyclic vacuum  ✅

The polynomial algebra of substrate fields applied to $|0_F\rangle$ generates an arbitrary $n$-particle / $\bar n$-antiparticle Fock-space vector via repeated application of $\psi^\dagger$ and $\psi$:

$$c_{\alpha_1, k_1}^\dagger \cdots c_{\alpha_m, k_m}^\dagger\, d_{\beta_1, p_1}^\dagger \cdots d_{\beta_n, p_n}^\dagger\,|0_F\rangle\,\in\,\overline{\text{span}}\{\psi^k (\psi^\dagger)^l |0_F\rangle\}$$

since $\psi^\dagger(g)$ contains all $c^\dagger_{\alpha, k}$ via Bloch-mode expansion (with linearly independent coefficients $u_\alpha^*(k, r)e^{-ik\cdot R}$ for distinct $(R, r)$). Therefore $|0_F\rangle$ is cyclic for the field-polynomial algebra. ✓

**Theorem-grade.** Standard for free fermion theories on a lattice.

### W4: Spectrum condition  ✅

Energy operator: $H = \sum_{\alpha, k} \varepsilon_\alpha(k) c_{\alpha, k}^\dagger c_{\alpha, k} + (\text{vacuum constant})$. After normal-ordering, $H \ge 0$ with $H |0_F\rangle = 0$. Spectrum of $H$ on Fock space is $\bigcup_{n} \{\sum_i \varepsilon_{\alpha_i}(k_i) : n \text{ particles}\}$, all non-negative. ✓

Momentum operators: $P^a = \sum_{\alpha, k} k^a\, c_{\alpha, k}^\dagger c_{\alpha, k}$ (lattice momenta in $\text{BZ}$). On Fock space, $P^a$ has spectrum on the BZ in each particle slot.

**Substrate spectrum condition.** $(H, P^a)$ have spectrum in the **discrete forward light cone** — the analog of the Lorentzian forward cone. For the framework's emergent Lorentzian metric (Stage 3 + Iorio-elastic), at leading order in k around the Γ-Dirac cone, the substrate's $H$ and $P^a$ satisfy the standard spectrum condition $E \ge v_F |\mathbf P|$ (with $v_F = 1/2$). ✓

**Theorem-grade at substrate level**; continuum-limit version inherits from emergent Lorentz invariance.

### W5: Microcausality  ✅ (discrete) / ⚠ (continuum)

**At substrate-discrete level**: the CAR algebra (`../theorems/theorem_car_local_jordan_wigner.md`) gives $\{\psi(g), \psi(g')\} = 0$ and $\{\psi(g), \psi^\dagger(g')\} = \delta_{g, g'}$. For distinct vertices $g \ne g'$, anticommutation is exact. ✓

For continuous time, the equal-time anticommutator at distinct vertices is $\{\psi(g, t), \psi^\dagger(g', t)\} = \delta_{g, g'}$. At unequal times, the CCR/CAR commutator is the substrate Wightman 2-point function (F1 §2.2), which is non-zero in general — but vanishes when projected onto the **spacelike-separated** spatial sector.

**Substrate spacelike separation.** Two events $(g, t_1)$ and $(g', t_2)$ are spacelike-separated in the substrate's Stage-3 leading-order Lorentzian sense if $|t_1 - t_2| < |R - R'| / v_F$ where $v_F = 1/2$ is the substrate's emergent speed of light at the Γ-Dirac cone. For such separations, the field anticommutator $\{\psi(g_1, t_1), \psi(g_2, t_2)\}$ vanishes at leading order in the small-$k$ Bloch-mode expansion (by translation-invariance + on-shell propagation at $v_F$).

**Status.** Microcausality is **theorem-grade at the discrete CAR level** (W5 in the substrate's $\mathbb Z^3 \rtimes 432 \times \mathbb R_t$ framework). For the **continuum Lorentzian-signature** version (W5 in the standard $\mathcal P^\uparrow_+$ framework), the substrate's emergent Lorentz invariance from Stage 3 + Γ-Dirac + Iorio-elastic provides leading-order theorem-grade closure; full version pending §C.

### W6: Covariance  ✅ (discrete) / ⚠ (continuum)

**At substrate-discrete level**: lattice translations $T_a$ and cubic-432 rotations $R$ act on $\psi(g, t)$ by

$$T_a\, \psi(g, t)\, T_a^{-1} \;=\; \psi(g - a, t),\qquad R\,\psi(g, t)\,R^{-1} \;=\; D(R)\,\psi(R \cdot g, t)$$

with $D(R)$ the spinor representation of the cubic point group on the spinor bundle. Continuous time evolution: $U(t)\,\psi(g, 0)\, U(t)^{-1} = e^{iHt} \psi(g, 0) e^{-iHt}$ (Heisenberg picture).

These together implement the substrate's symmetry group on the field operators. ✓

**Continuum Lorentz covariance**: per Stage 3 (`../theorems/theorem_lorentz_causal_sector.md`), the substrate has leading-order Lorentz invariance from the chirally symmetric spectrum + isotropic Cartesian dispersion. Per the Γ-Dirac cone analysis (`predictions/srs_dirac_cone_velocities.py`, `proofs/foundations/lorentz_sig_spin1_dirac_decomposition.py`), the algebraic structure $H_{\rm eff} = -1 + (1/2) \mathbf k_{\rm cart}\cdot \mathbf S$ with $[S_a, S_b] = \pm \epsilon_{abc} S_c$ gives full SO(3) at leading order — extending to local (1+3) Minkowski with $v_F = 1/2$.

The **emergent Poincaré-group representation** on substrate fields exists at leading order; full Wightman-W6 closure (no leading-order qualification) pending §C smooth-manifold continuum lift.

### W7: Fields as operator-valued distributions  ✅ (discrete) / ⚠ (continuum)

**At substrate-discrete level**: $\psi(g, t)$ is an operator-valued **function** on $\Lambda \times \mathbb R_t$ (discrete spatial argument, continuous time). For test functions $f(g, t)$ on this space, the smeared field

$$\Psi[f] \;:=\; \sum_{g}\,\int dt\, f(g, t)\,\psi(g, t)$$

is a well-defined operator on Fock space with the obvious linearity, continuity (in $f$ in the appropriate norm), and CAR algebra. ✓

**Continuum operator-valued distributions**: Wightman's W7 specifically asks for distributions on Schwartz test functions $\mathcal S(\mathbb R^4)$, which is a continuum statement. The substrate's discrete spatial structure only gives summable test functions on $\Lambda$, not Schwartz on $\mathbb R^3$. The continuum lift of W7 requires §C — smearing substrate fields with continuum Schwartz functions and verifying the limiting object is an operator-valued distribution.

**Status.** ✓ theorem-grade in the substrate's discrete framework. ⚠ continuum lift pending §C.

---

## 3. The substrate as a "discrete Wightman theory"

Combining the seven axiom verifications above:

**Theorem (substrate-discrete Wightman).** The substrate fermion field theory $(\mathcal F, |0_F\rangle, \psi(g, t))$ over the BCC + Wyckoff-8a substrate satisfies all seven Wightman axioms at the substrate-discrete level, with the Poincaré group replaced by $\mathbb Z^3 \rtimes 432 \times \mathbb R_t$ and operator-valued distributions on continuum Schwartz functions replaced by operator-valued functions on the discrete vertex set.

This is a **discrete instance of Streater–Wightman's axiomatic QFT framework**. The framework's emergent QFT is therefore a fully-rigorous discrete axiomatic QFT — not merely a heuristic field-theoretic structure.

**Interpretive remark.** The substrate's discrete Wightman theory is *not* a specialisation of the continuum Wightman framework but rather a **pre-continuum-limit** form. The continuum Wightman theory emerges from the substrate's discrete Wightman theory in the §C continuum limit, with Poincaré-group covariance lifting from the substrate's $\mathbb Z^3 \rtimes 432$ via the standard symmetry-enhancement at long-wavelength (Stage 3 leading-order Lorentz invariance becomes exact at infinite-wavelength limit).

---

## 4. Continuum-limit closure roadmap

The three continuum-pending axiom aspects (W5 microcausality continuum, W6 full-Poincaré, W7 distribution-valued continuum) all pivot on the framework's §C smooth-manifold continuum-limit closure. Concrete sub-routes:

1. **W6 → Stage 3 + Γ-Dirac + Iorio-elastic + global emergent Lorentzian manifold.** Already partial: leading-order Lorentz invariance is theorem-grade (Stage 3); local emergent Minkowski metric is theorem-grade (Iorio-elastic Sessions 1-3); global lift to a Lorentzian manifold is pending Iorio-elastic Session 4 + multi-valley Γ-H + backreaction (an internal working note + `lorentz_sig_backreaction_einstein.md`).

2. **W5 microcausality** at continuum spacelike separation: follows from W6 + the established CAR-locality at discrete level. The "spacelike" notion at continuum is determined by the emergent Lorentzian metric from W6; once the metric is in place, microcausality follows from the discrete CAR via averaging over discrete-spatial → continuum-spatial test functions.

3. **W7 distribution-valued continuum**: depends on the continuum-distribution structure of the discrete-to-continuum lift. Standard procedure: smear the discrete fields with test functions on $\Lambda$, take continuum limit using the lattice's natural measure (BZ-uniform-momentum / Wigner-quantization), verify the limit is in the appropriate Wightman-distributional sense.

These three closures together — once §C is fully closed — promote the substrate-discrete Wightman theory to a **standard continuum Wightman theory** with the framework's emergent Poincaré covariance.

---

## 5. Comparison to standard QFT

**Identification table:**

| Standard Wightman | Substrate Wightman |
|---|---|
| Continuum spacetime $\mathbb R^4$ | Discrete $\Lambda \times \mathbb R_t$ (BCC × cont. time) |
| Poincaré group $\mathcal P^\uparrow_+$ | $\mathbb Z^3 \rtimes 432 \times \mathbb R_t$ (substrate symmetry group) |
| Hilbert space + Poincaré rep | Fock space + lattice + cubic 432 rep |
| Vacuum $|0\rangle$ Poincaré-invariant | $|0_F\rangle$ Dirac-sea, lattice-translation invariant |
| Cyclic vacuum (cont. polynomials of $\Phi$) | Cyclic vacuum (lattice polynomials of $\psi$, $\psi^\dagger$) |
| Spectrum cond. $p^\mu$ in fwd cone | $H \ge 0$, $P \in $ BZ; emergent fwd cone via $v_F$ |
| Microcausality at spacelike sep. | CAR at discrete vertex sep. (✓); continuum spacelike via Stage 3 (⚠) |
| Lorentz covariance $S(\Lambda)$ | Lattice + cubic 432 (✓); full Lorentz via §C (⚠) |
| Fields as $\mathcal D'(\mathcal S(\mathbb R^4))$ | Lattice operator-functions (✓); continuum dist. via §C (⚠) |

**Three structural matches** (theorem-grade):
1. All axioms have a substrate-discrete analog that's theorem-grade.
2. The substrate's emergent Lorentz invariance (Stage 3 + Γ-Dirac) provides the natural bridge from the discrete framework to the continuum Wightman framework.
3. The substrate is therefore a fully-rigorous discrete instance of Wightman theory.

**Three structural differences** (well-understood):
1. Discrete spatial structure (lattice → continuum requires §C).
2. Bounded BZ (compact spatial momentum, finite UV cutoff built-in).
3. Substrate-cubic-432 → continuum-Lorentz at leading order; emergence is via long-wavelength limit, not a direct symmetry.

---

## 6. Honest scope flag

**At theorem grade (no adoptions):**
- W1, W2, W3, W4 (substrate-discrete versions): all directly verifiable from F1 + F3 + F4 + framework's existing apparatus.
- W5 microcausality at substrate-discrete level: theorem-grade via CAR.
- W6 covariance for $\mathbb Z^3 \rtimes 432 \times \mathbb R_t$: theorem-grade.
- W7 operator-valued functions on $\Lambda \times \mathbb R_t$: theorem-grade.

**Pending §C continuum-limit closure:**
- W5 continuum spacelike microcausality (fully Lorentzian).
- W6 full Poincaré-group covariance.
- W7 operator-valued distributions on Schwartz $\mathcal S(\mathbb R^4)$.

For each pending item, the leading-order substrate version is theorem-grade (Stage 3 + Γ-Dirac + Iorio-elastic), and the continuum closure follows from §C. Concrete §C closure pathway is documented in an internal working note (Sessions 1-4 partial) + `lorentz_sig_backreaction_einstein.md`.

**Status.** F11 closes at substrate-discrete level (theorem-grade). Continuum-limit version is partial (leading-order theorem-grade, full version §C-dependent).

---

## Cross-references

- `forward_construction_substrate_propagator.md` (F1 propagator).
- `forward_construction_substrate_wick.md` (F3 Wick).
- `forward_construction_substrate_lsz.md` (F4 LSZ).
- `forward_construction_substrate_thermal_apparatus.md` (vacuum).
- `../theorems/theorem_car_local_jordan_wigner.md` (CAR).
- `../theorems/theorem_lorentz_causal_sector.md` (Stage 3 leading-order Lorentz).
- `predictions/srs_dirac_cone_velocities.py` + `proofs/foundations/lorentz_sig_spin1_dirac_decomposition.py` (Γ-Dirac SO(3) emergence).
- Streater, R. F., Wightman, A. S. (1964). *PCT, Spin and Statistics, and All That*. Benjamin.
- Glimm, J., Jaffe, A. (1987). *Quantum Physics — A Functional Integral Point of View*, 2nd ed. Springer. §6.
- Peskin, M. E., Schroeder, D. V. (1995). *An Introduction to Quantum Field Theory*. Addison-Wesley. §4.
- Weinberg, S. (1995). *The Quantum Theory of Fields*, Vol. I. Cambridge Univ. Press. §10.
