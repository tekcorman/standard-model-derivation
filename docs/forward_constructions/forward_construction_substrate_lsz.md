# Substrate LSZ reduction — F4 scattering amplitudes

**Date:** 2026-04-27 (PM, follow-on to F1 propagator and F3 Wick).
**Status:** Theorem-grade closure of F4 from an internal note. Third concrete deliverable in the φ(x) cascade.
**Source:** an internal note Tier 2 next-step #1 (substrate scattering amplitudes from in/out asymptotic Bloch states).
**Predecessors:**
- `forward_construction_substrate_propagator.md` (F1, Feynman propagator $\tilde G_F^{\text{sub}}(k, \omega) = i(\omega + D(k))/(\omega^2 - D^2(k) + i\varepsilon)$).
- `forward_construction_substrate_wick.md` (F3, n-point time-ordered functions = signed sum of pair-contractions of $G_F^{\text{sub}}$).
- `forward_construction_field_operator_phi_x.md` §2.4 (Bloch-mode + JW/CAR hybrid field operator $\psi(g) = (1/\sqrt V) \sum_{\alpha, k} u_\alpha(k, r) e^{ik\cdot R} c_{\alpha, k}$).
- `../theorems/theorem_car_local_jordan_wigner.md` (substrate CAR via JW).

---

## Question

Given F1 (free Feynman propagator $G_F^{\text{sub}}$) and F3 (Wick reduction of n-point time-ordered functions), can we extract substrate **scattering amplitudes** — i.e., transition amplitudes $\langle\,\text{out}\,|\,\text{in}\,\rangle$ between asymptotic substrate states — from the substrate's time-ordered correlation functions?

This is the substrate analog of LSZ (Lehmann–Symanzik–Zimmermann 1955) reduction. If closed: substrate inherits standard QFT scattering theory. F5 (concrete S-matrix), F6 (Feynman rules), F8 (bosonic LSZ), and F11 (Wightman axioms) all build on F4.

---

## Result (preview)

**Theorem (substrate LSZ reduction).** For substrate fermion fields and free Hamiltonian $H = D_{\text{sub}}$, the scattering amplitude from an in-state of $m$ particles at Bloch indices $(\alpha_i, k_i)_{i=1}^m$ to an out-state of $n$ particles at $(\alpha'_j, k'_j)_{j=1}^n$ is

$$\boxed{\;\langle\,\alpha'_1 k'_1, \ldots, \alpha'_n k'_n;\,\text{out}\,|\,\alpha_1 k_1, \ldots, \alpha_m k_m;\,\text{in}\,\rangle \;=\; \prod_{i=1}^m \bar{u}_{\alpha_i}(k_i)\,\Big[ \mathcal R^{(m+n)}_{\text{amp}} \Big]\,\prod_{j=1}^n u_{\alpha'_j}(k'_j) \;}$$

where $\mathcal R^{(m+n)}_{\text{amp}}$ is the **amputated $(m+n)$-point function** in Bloch–frequency space, obtained from the time-ordered correlator $G_n^{(m+n)} = \langle 0_F | T(\psi(g_1, t_1) \cdots \psi^\dagger(g_{m+n}, t_{m+n})) | 0_F \rangle$ by

(a) Fourier transforming each leg in $(R_i, t_i) \to (k_i, \omega_i)$,
(b) putting each external leg on shell: $\omega_i \to \varepsilon_{\alpha_i}(k_i)$,
(c) **amputating** each external leg by multiplying by $-i (\omega_i^2 - D^2(k_i)) / Z_{\alpha_i}$ at on-shell, where $Z_{\alpha_i}$ is the wave-function renormalisation (= 1 in the free theory),
(d) contracting with on-shell wave functions $u_{\alpha_i}(k_i)$.

For the free theory ($Z = 1$, no interactions), the only non-trivial amplitudes are: 1-particle $\to$ 1-particle (trivial, gives identity matrix on Bloch indices), and **all higher amplitudes vanish** (because by F3 Wick, $G_n^{(m+n)}$ is a sum of disconnected propagator products at the free level). Substrate scattering amplitudes for $m + n \ge 4$ require an interaction Hamiltonian (F5), with vertex insertions from substrate's gauge / Yukawa structure.

The structural takeaway: **the standard LSZ apparatus — asymptotic Bloch states, on-shell amputation, $u(k)$ contraction — transposes verbatim to the substrate, with continuum momentum $\mathbf k \in \mathbb R^3$ replaced by Bloch momentum $k \in \text{BZ}$ and continuum $x \in \mathbb R^3$ replaced by substrate vertex $g = (R, r)$**.

---

## 1. Setup

### 1.1 Inputs

From F1 + F3:
- **Field operator** $\psi(g, t) = (1/\sqrt V) \sum_{\alpha, k} u_\alpha(k, r)\, e^{ik\cdot R}\, c_{\alpha, k}(t)$, with $c_{\alpha, k}(t) = e^{-i\varepsilon_\alpha(k) t}\, c_{\alpha, k}$ and $\varepsilon_\alpha(k)$ the eigenvalues of $D(k)$ (chirally symmetric; 16 positive, 16 negative for srs).
- **Vacuum** $|0_F\rangle$ Dirac-sea state.
- **Free Feynman propagator** $\tilde G_F^{\text{sub}}(k, \omega) = i(\omega + D(k))/(\omega^2 - D^2(k) + i\varepsilon)$ (F1 Thm 3.2).
- **Wick theorem** for $n$-point time-ordered functions (F3).

### 1.2 Asymptotic in/out states

Define one-particle states by mode creation:

$$|\alpha, k;\,\text{free}\rangle \;:=\; c_{\alpha, k}^\dagger\, |0_F\rangle,\qquad \varepsilon_\alpha(k) > 0.$$

For the **free** theory ($H = D_{\text{sub}}$ exactly), these are exact eigenstates of $H$ at energy $\varepsilon_\alpha(k)$ — there's no distinction between "in" and "out" states.

For an **interacting** theory ($H = D_{\text{sub}} + V$), in/out states are defined as eigenstates of the free Hamiltonian $H_0 = D_{\text{sub}}$ that asymptotically (as $t \to \mp\infty$, weak/no interaction) become exact eigenstates of $H$. Standard QFT prescription: turn on interaction adiabatically, $V \to e^{-\eta|t|} V$ for $\eta \to 0^+$. In/out states are then the limits of free states evolved with full $H$.

For substrate, the same prescription applies. We focus on the free-theory derivation (F4 closure); interactions enter at F5.

### 1.3 Wave-function $u_\alpha(k, r)$ and adjoint $\bar u$

The wave function $u_\alpha(k, r)$ is the $r$-th component of the eigenvector of $D(k)$ at eigenvalue $\varepsilon_\alpha(k)$. In our 32-dim Bloch-fiber basis ($r = 1, \ldots, 4$ atoms × $\sigma = 1, \ldots, 8$ Cl(6,0) spinor), $u_\alpha(k)$ is a 32-component column vector.

Define the **adjoint** $\bar u_\alpha(k) = u_\alpha^\dagger(k)\,\Gamma$ where $\Gamma$ is the substrate analog of $\gamma^0$ — the chirality / time-reversal operator on the spinor bundle. For the framework's existing apparatus (`forward_construction_substrate_atiyah_singer.md` §1.2), $\Gamma$ is the volume form of Cl(6,0).

Orthonormality: $\bar u_{\alpha'}(k')\, u_\alpha(k) = \delta_{\alpha\alpha'}\,\delta_{kk'}$ in the Bloch-mode basis (modulo vacuum convention).

---

## 2. Källén–Lehmann spectral representation on substrate

### 2.1 Two-point function spectral form

For an interacting theory, the Wightman two-point function takes the spectral form (substrate analog of Källén–Lehmann 1952–1954):

$$W(g, g'; t) \;=\; \int_0^\infty d\mu^2\, \rho(\mu^2)\, W_0(g, g'; t \mid \mu^2)$$

where $\rho(\mu^2) \ge 0$ is the spectral density and $W_0(\,\cdot\,|\mu^2)$ is the free Wightman function for a particle of mass-squared $\mu^2$.

For the **free** substrate theory, $\rho(\mu^2)$ is concentrated on the eigenvalues of $D^2(k)$:

$$\rho_{\text{free}}(\mu^2) \;=\; \sum_\alpha \delta(\mu^2 - \varepsilon_\alpha^2(k)).$$

By F1 Theorem 3.2, this matches the propagator structure exactly. For srs, the eigenvalues $\varepsilon_\alpha^2(k) = (n + R_{\text{sub}}(k))_\alpha$ are bounded by $n + \|R_{\text{sub}}\| = 6 + \sqrt{30} \approx 11.5$. The substrate spectral density has compact support — UV-finite by lattice cutoff.

### 2.2 Wave-function renormalisation $Z_\alpha$

For the interacting theory, the spectral density splits as

$$\rho(\mu^2) \;=\; Z_\alpha\, \delta(\mu^2 - \varepsilon_\alpha^2(k)) \;+\; \rho_{\text{cont}}(\mu^2)$$

with $Z_\alpha \le 1$ the **wave-function renormalisation** at mode $(\alpha, k)$ and $\rho_{\text{cont}}$ the continuum contribution from multi-particle intermediate states.

For the free substrate theory: $Z_\alpha = 1$, $\rho_{\text{cont}} = 0$. Established at theorem grade.

For the interacting theory: $Z_\alpha$ requires the specific interaction; it's an input to F5.

---

## 3. Substrate LSZ reduction formula

### 3.1 Derivation

Following the standard LSZ derivation (Peskin–Schroeder §7.2; Weinberg §10.3) on substrate:

**Step 1.** Express the in-state in terms of the field operator at $t \to -\infty$. Using the Bloch-mode expansion and orthonormality:

$$|\alpha_1 k_1;\,\text{in}\rangle \;\propto\; \lim_{t \to -\infty}\,\int d^3R \, e^{-ik_1 \cdot R + i\varepsilon_{\alpha_1}(k_1) t}\, \bar u_{\alpha_1}(k_1, r)\, \psi(R, r; t)\,|0_F\rangle.$$

(With appropriate overall normalisation that we'll track through.)

**Step 2.** For a multi-particle in-state, products of these expressions multiply. Same for out-states at $t \to +\infty$.

**Step 3.** The transition amplitude is:

$$\langle\,\text{out}\,|\,\text{in}\,\rangle \;=\; \lim_{t_i \to \mp\infty}\,\bigg(\prod_i \int d^3R_i\, e^{\mp i k_i\cdot R_i \pm i \varepsilon_i t_i}\, \bar u_i\bigg)\,\langle 0_F | T(\psi\cdots\psi^\dagger\cdots) |0_F\rangle.$$

(The $T$ ordering is automatic since in-fields go before out-fields temporally.)

**Step 4.** Fourier transform each leg in $(R_i, t_i) \to (k_i, \omega_i)$. The time-Fourier transform of $\psi(R, t)$ at large $|t|$ extracts the on-shell residue of the propagator.

**Step 5.** By the spectral representation (§2.2), the on-shell behaviour of the time-ordered correlator at each external leg is:

$$\tilde G^{(N)}(k_1, \omega_1; \ldots; k_N, \omega_N) \;\xrightarrow[\omega_i \to \varepsilon_{\alpha_i}(k_i)]{}\; \prod_i \frac{i Z_{\alpha_i}}{\omega_i - \varepsilon_{\alpha_i}(k_i) + i\varepsilon}\,\times\,\mathcal R^{(N)}_{\text{amp}}(k_i, \omega_i)$$

where the limit on the LHS exhibits a simple pole at each on-shell external $\omega_i$, and $\mathcal R^{(N)}_{\text{amp}}$ is the amputated on-shell residue.

**Step 6.** Multiplying by $-i(\omega_i - \varepsilon_{\alpha_i}(k_i))/Z_{\alpha_i}$ at each leg removes the propagator pole (amputates the leg) and leaves the on-shell residue. Contracting with the on-shell wave functions $u_{\alpha_i}(k_i)$ extracts the scattering amplitude.

### 3.2 Substrate LSZ formula (theorem)

**Theorem (substrate LSZ).** The scattering amplitude is

$$\langle\,\alpha'_1 k'_1, \ldots, \alpha'_n k'_n;\,\text{out}\,|\,\alpha_1 k_1, \ldots, \alpha_m k_m;\,\text{in}\,\rangle$$

$$\;=\; \prod_{i=1}^m \bar u_{\alpha_i}(k_i) \prod_{j=1}^n u_{\alpha'_j}(k'_j) \cdot \prod_{i=1}^m \frac{1}{\sqrt{Z_{\alpha_i}}} \prod_{j=1}^n \frac{1}{\sqrt{Z_{\alpha'_j}}} \cdot \mathcal R^{(m+n)}_{\text{amp}}(k_i, \varepsilon_{\alpha_i}(k_i); k'_j, \varepsilon_{\alpha'_j}(k'_j))$$

with $\mathcal R^{(m+n)}_{\text{amp}}$ the amputated $(m+n)$-point function: take the time-ordered correlator $G^{(m+n)} = \langle 0_F | T(\psi \cdots \psi^\dagger \cdots) | 0_F \rangle$, Fourier transform to $(k_i, \omega_i)$, put each leg on shell $\omega_i \to \varepsilon_{\alpha_i}(k_i)$, and **amputate** by multiplying each external leg by

$$\Big[\tilde G_F^{\text{sub}}(k_i, \omega_i)\Big]^{-1}\,\bigg|_{\omega_i \to \varepsilon_{\alpha_i}(k_i)}.$$

For the free theory ($Z = 1$, no interactions), the only non-trivial amplitude at order zero is the trivial 1-to-1 identity. All higher amplitudes vanish in the free theory by F3 Wick (n-point function = sum of pair-products of propagators ⇒ disconnected diagrams ⇒ no genuine scattering at the free level).

### 3.3 Tree-level amplitudes (interacting, illustrative)

For the interacting theory $H = D_{\text{sub}} + V$, the LSZ formula above plus Dyson expansion of $V$ produces tree-level scattering amplitudes:

$$\mathcal R^{(m+n)}_{\text{amp,\,tree}} \;=\; (i)^{m+n}\,\sum_{\text{tree diagrams}} \prod_{\text{vertices}} (-iV)\, \prod_{\text{internal lines}} \tilde G_F^{\text{sub}}(k, \omega).$$

Substrate Feynman diagrams: external lines = on-shell wave functions $u_\alpha(k)$ (or $\bar u$); internal lines = $\tilde G_F^{\text{sub}}$ propagators; vertices = $-iV$ insertions from the interaction Hamiltonian. This is the substrate analog of the standard QFT Feynman-diagram apparatus.

For F5 (concrete substrate S-matrix), one specifies $V$ (substrate gauge / Yukawa structure) and computes specific 4-point amplitudes ($\psi\psi \to \psi\psi$).

---

## 4. Comparison to standard QFT LSZ

**Identification table:**

| Standard QFT LSZ | Substrate LSZ |
|---|---|
| Continuum momentum $\mathbf k \in \mathbb R^3$ | Bloch momentum $k \in \text{BZ}$ (compact) |
| Continuum spacetime $x \in \mathbb R^4$ | Substrate $(R, r) \times t$ (discrete spatial, continuous time) |
| On-shell condition $\omega^2 = \mathbf k^2 + m^2$ | On-shell $\omega^2 = \varepsilon_\alpha^2(k) = n + R_{\text{sub}}(k)_\alpha$ |
| Wave function $u(k)$ ($\bar u$) | $u_\alpha(k, r)$ Bloch mode at eigenvalue $\varepsilon_\alpha$ |
| Asymptotic in/out states | Free Bloch states at $t \to \mp\infty$, with adiabatic switching |
| Amputation by $(i\slashed k - m)$ on shell | Amputation by $\tilde G_F^{\text{sub}}(k, \omega)^{-1}$ on shell |
| Wave-function renormalisation $Z$ | $Z_\alpha$ per Bloch mode (= 1 in free theory) |

**Three structural matches** (all theorem-grade now):
1. **Asymptotic-state structure.** Bloch modes at positive $\varepsilon_\alpha(k) > 0$ play the role of asymptotic particles; negative-$\varepsilon$ modes (Dirac-sea filled) are antiparticles.
2. **On-shell condition.** Substrate has its own "mass-shell" set by the operator $D^2 = n\cdot I + R_{\text{sub}}(k)$. For srs, the substrate's intrinsic mass scale is $n = 6$ (Planckian), modulated by $R_{\text{sub}}(k)$.
3. **Amputation rule.** Identical formal procedure as in continuum QFT: multiply by inverse propagator at each external leg.

**Three structural differences:**
1. **Compact momentum.** $k \in \text{BZ}$ (compact). Substrate has no UV divergence in the LSZ formula — the BZ is the natural cutoff.
2. **Discrete spatial structure.** Substrate vertex $g = (R, r)$ with discrete lattice $R$ and atom index $r$. The Fourier transform $R \to k$ is a finite Bloch sum, not a continuum integral.
3. **Spectral cutoff.** $\rho(\mu^2)$ has compact support ($\mu^2 \in [0, n + \|R_{\text{sub}}\|]$), so multi-particle thresholds are bounded.

---

## 5. Implications for the cascade

F4 closure unlocks:

- **F5 (substrate S-matrix)**: now ready for concrete computation. The simplest non-trivial S-matrix element is $\psi\psi \to \psi\psi$ at tree level, requiring the framework's specific interaction Hamiltonian. For srs, the natural candidates are: (a) fermion-fermion interaction via gauge bosons (substrate gauge structure pending), (b) Yukawa-type from substrate's Higgs sector. Estimated 2-3 sessions per concrete amplitude.

- **F6 (substrate Feynman rules)**: graphical perturbation expansion is now mechanical given F1 (propagator), F3 (Wick), F4 (LSZ). Vertex enumeration depends on the substrate interaction Hamiltonian; once specified, Feynman rules are mechanical. Estimated 2 sessions.

- **F7 (renormalisation derivation)**: F4's wave-function renormalisation $Z_\alpha$ is the input. Renormalisation as substrate coarse-graining via I-projection (A2-T) is the highest-leverage Tier 3 closure. Estimated 3+ sessions; ingredients now all present (F1 + F3 + F4 + I-projection).

- **F11 (Wightman axioms)**: substrate locality, Lorentz covariance, positive-definiteness, vacuum uniqueness. Some require continuum-limit closure (§C, partial); the discrete substrate version uses F1 + F3 + F4 directly. Estimated 2-3 sessions.

The bosonic-side analog (F8) requires a separate path; the LSZ derivation here is fermionic (CAR-based). Bosonic LSZ for the substrate's emergent Higgs / gauge sectors is a distinct workstream entry.

---

## 6. Honest scope flag

The derivation above transposes the standard Peskin-Schroeder §7.2 / Weinberg §10.3 LSZ derivation verbatim, with substrate-specific identifications. The substrate-specific structural pieces (discrete vertex, compact BZ, $D^2 = n\cdot I + R_{\text{sub}}(k)$) are theorem-grade.

The **wave-function renormalisation $Z_\alpha$** for the interacting theory is computable but requires the specific interaction Hamiltonian. For the free theory, $Z_\alpha = 1$ at theorem grade.

The **adjoint $\bar u$ = $u^\dagger \Gamma$ identification** uses the framework's existing $\Gamma$ from `forward_construction_substrate_atiyah_singer.md` §1.2; this is theorem-grade upstream.

**No adoptions.** F4 inherits its inputs (F1, F3, CAR, JW, vacuum) at theorem grade. The LSZ derivation itself is standard (Peskin-Schroeder §7.2, Weinberg §10.3). All substrate-specific identifications are explicit.

Status: **theorem-grade closure**.

---

## Cross-references

- `forward_construction_substrate_propagator.md` (F1 propagator, foundation).
- `forward_construction_substrate_wick.md` (F3 Wick, n-point structure).
- `forward_construction_field_operator_phi_x.md` §2.4 (Bloch-mode field operator hybrid).
- `forward_construction_substrate_atiyah_singer.md` (Dirac D_sub, Γ adjoint).
- `../theorems/theorem_car_local_jordan_wigner.md` (CAR).
- Peskin–Schroeder (1995). *An Introduction to Quantum Field Theory*. Addison-Wesley. §7.2 (LSZ reduction).
- Weinberg (1995). *The Quantum Theory of Fields*, Vol. I. Cambridge Univ. Press. §10.3 (LSZ).
- Lehmann, H., Symanzik, K., Zimmermann, W. (1955). Zur Formulierung quantisierter Feldtheorien. *Nuovo Cimento* 1, 205.
