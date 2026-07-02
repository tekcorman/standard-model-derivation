# Derivation of the dark extraction map

**NOTE (post-2026-04-26 demotion):** A2 and A3 are derived theorems; structural slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure chain referenced here is preserved; only the axiomatic-status labels change. G.1 and G.5 are DERIVED via CDP 2011 Theorem 25 (predictions/observer_hilbert_space.py). The Hilbert-space side of Feshbach/uniform-Q-density is therefore no longer assumed; their derivation from A1 + A2-T + A3-T remains separately load-bearing.

## Abstract

We derive which component of the dark sector self-energy $\Sigma(h) = \alpha_1/h$ each physical observable couples to, using $C_3 \times$ parity representation theory at the P-point of the srs lattice. The result: three classes of dark correction, each determined by the observable's quantum numbers. No fitting to observation is used — the classification follows from $C_3$ selection rules and the Hermitian decomposition of the walker operator.

## Framework axioms invoked

None beyond upstream:
- $h = (\sqrt{3} + i\sqrt{5})/2$ (from `predictions/h_walker_eigenvalue.py`)
- $\alpha_1 = (2/3)^8$ (from `predictions/alpha_1.py`)
- $\Sigma(h) = \alpha_1/h$ (derived from MDL + Feshbach in `dark_correction_theorem_2026-04-14.md` §4a: uniform Q-space density from MDL maximality, contour integral gives $\Sigma = \alpha_1/h$)

## Derivation

### Step 1: Self-energy structure under $C_3$

At the P-point, the walker eigenvalue $h$ transforms as the $\omega$ irrep of $C_3$ (the generation quantum number — proven in `proofs/foundations/srs_generation_c3.py`). Its conjugate $h^*$ transforms as $\omega^2 = \bar{\omega}$.

The self-energy:

$$\Sigma(h) = \frac{\alpha_1}{h} = \frac{\alpha_1}{|h|^2} h^* = \frac{\alpha_1}{2} h^* \tag{1}$$

Since $h^*$ transforms as $\omega^2$:

$$\boxed{\Sigma \text{ transforms as } \omega^2 \text{ under } C_3} \tag{2}$$

$\Sigma$ is **purely off-diagonal** in generation space. This is a consequence of the self-energy being proportional to $h^*$ (the complex conjugate walker). This step is the $C_3$ representation assignment of a product of known quantities — standard representation theory (Serre, *Linear Representations of Finite Groups*, 1977, §2.4).

### Step 2: Hermitian decomposition of the walker operator

The walker's velocity operator $B$ at the P-point has eigenvalue $h = \text{Re}(h) + i\,\text{Im}(h)$. The Hermitian decomposition:

$$B = B_{\text{sym}} + i\,B_{\text{anti}} \tag{3}$$

where $B_{\text{sym}} = (B + B^\dagger)/2$ has eigenvalue $\text{Re}(h) = \sqrt{3}/2$ and $B_{\text{anti}} = (B - B^\dagger)/(2i)$ has eigenvalue $\text{Im}(h) = \sqrt{5}/2$.

Under parity (the map $h \to h^*$, equivalently $B_{\text{anti}} \to -B_{\text{anti}}$):
- $B_{\text{sym}}$ is **parity-even** (invariant)
- $B_{\text{anti}}$ is **parity-odd** (sign-reverses)

This is the standard Hermitian decomposition of a complex operator into self-adjoint and anti-self-adjoint parts (Halmos, *Finite-Dimensional Vector Spaces*, §73). The parity assignment follows from $B_{\text{anti}}$ being purely imaginary.

### Step 3: Classification by $C_3 \times$ parity quantum numbers

Each observable has definite $C_3$ quantum number (which generation structure it probes) and parity (whether it involves $B$ or $B^\dagger B$). These quantum numbers determine which component of $\Sigma$ the observable couples to.

#### Class 1: Amplitude (off-diagonal, 1-point)

**Observables:** Walk amplitudes between different generations — $V_{us}$, $m_{\nu_2}$, $m_{\nu_3}$ (via seesaw, which crosses generations).

**$C_3$ quantum number:** $\omega^n$ with $n \neq 0$ (off-diagonal, generation-changing).

**Coupling:** $\Sigma$ transforms as $\omega^2$ (Step 1), which IS off-diagonal. The observable couples to $\Sigma$ at **first order** (direct insertion). The $C_3$ selection rule allows this because the product (observable's $C_3$ charge) $\times$ ($\Sigma$'s $C_3$ charge) can equal the trivial representation.

**Extraction:** The observable measures a walk **magnitude** $|A|$. The magnitude of the self-energy correction is:

$$\delta|A| / |A| = |\text{Im}[\Sigma(h)]| \tag{4}$$

Why the imaginary part? The walk amplitude $A \propto h^d$ has a definite phase. The self-energy $\Sigma = (\alpha_1/2)h^*$ has both real and imaginary components. The real part of $\Sigma$ shifts the phase of $A$ (not its magnitude). The imaginary part of $\Sigma$ adds **in quadrature** to $A$, changing the magnitude. For a parity-odd observable (generation-changing), the coupling is through the parity-odd channel, selecting $\text{Im}(\Sigma)$.

$$|\text{Im}[\Sigma(h)]| = \frac{\alpha_1 \,|\text{Im}(h)|}{|h|^2} = \frac{\alpha_1 \sqrt{5}/2}{2} = \frac{\sqrt{5}}{4}\,\alpha_1 \tag{5}$$

**Coefficient: $\sqrt{5}/4 \approx 0.559$**

This is equation (5) — arithmetic from $h$ and $|h|^2$. The selection of Im (not Re) follows from the parity of the observable: generation-changing observables are parity-odd, coupling to the parity-odd component of $\Sigma$.

#### Class 2: Mass² (diagonal, 2-point)

**Observables:** Mixing angles from mass-matrix diagonalization — $\theta_{23}$.

**$C_3$ quantum number:** trivial (diagonal, generation-preserving). A mixing angle is a ratio of mass eigenvalues; the ratio is $C_3$-invariant.

**Coupling:** $\Sigma$ transforms as $\omega^2$ (off-diagonal). A single insertion of $\Sigma$ cannot contribute to a $C_3$-trivial observable (selection rule: $\omega^2 \neq 1$). The leading contribution is through the **mass matrix** $M^2 \propto B^\dagger B$, where the perturbation enters via the Hermitian channels of $B$.

The mass-matrix perturbation in the $\nu_2$-$\nu_3$ block (which is exactly degenerate at TBM):

$$\delta M^2 = \alpha_1 \left[\varepsilon_{\text{Re}}^2\,\sigma_z + \varepsilon_{\text{Im}}^2\,\sigma_x\right] \tag{6}$$

where:
- $\sigma_z$ is the diagonal (parity-even) Pauli matrix, with strength $\varepsilon_{\text{Re}}^2 = \text{Re}(h)^2 \cdot b_0$ from the parity-even channel of $B_{\text{sym}}$
- $\sigma_x$ is the off-diagonal (parity-odd) Pauli matrix, with strength $\varepsilon_{\text{Im}}^2 = \text{Im}(h)^2$ from the parity-odd channel of $B_{\text{anti}}$
- $b_0 = 1/2$ is the TBM off-diagonal normalization, fixed by the $C_3$ irrep structure at the P-point: the three $C_3$ eigenstates at a trivalent vertex have overlap $\langle\omega|\omega^2\rangle = 1/\sqrt{k^*}$, giving $b_0 = 1/k^* \cdot k^*/2 = 1/2$. (See `proofs/foundations/srs_generation_c3.py` for the $C_3$ decomposition.)

The degenerate perturbation theory on $M^2 = m^2 I + \delta M^2$ gives (standard 2×2 diagonalization — Sakurai, *Modern QM*, §5.2):

$$\Delta\theta_{23} = \frac{\varepsilon_{\text{Im}}^2}{2\,\varepsilon_{\text{Re}}^2}\,\alpha_1 = \frac{\text{Im}(h)^2}{2 \cdot \text{Re}(h)^2 \cdot (1/2)}\,\alpha_1 = \frac{\text{Im}(h)^2}{\text{Re}(h)^2}\,\alpha_1 = \tan^2(\arg h)\,\alpha_1 \tag{7}$$

**Coefficient: $\tan^2(\arg h) = 5/3 \approx 1.667$**

Every factor is traced:
- $\text{Im}(h)^2 = 5/4$: parity-odd channel strength (from $B_{\text{anti}}$ eigenvalue)
- $\text{Re}(h)^2 = 3/4$: parity-even channel strength (from $B_{\text{sym}}$ eigenvalue)
- $b_0 = 1/2$: TBM normalization (from $C_3$ irrep structure)
- The combination $(5/4)/(2 \cdot 3/4 \cdot 1/2) = (5/4)/(3/4) = 5/3$

#### Class 3: Edge-local ($C_3$-symmetric vertex)

**Observables:** Quantities measured at a $C_3$-symmetric vertex — $\theta_{13}$, $V_{cb}$ (commensurate detour at a vertex).

**Key property:** At a $C_3$-symmetric vertex, the parity-mixing operator $\sigma_x$ has:

$$\text{Tr}(\sigma_x) = 0 \tag{8}$$

because the three $C_3$ images of $\sigma_x$ cancel: $\sigma_x + \omega\sigma_x\omega^{-1} + \omega^2\sigma_x\omega^{-2} = 0$ (the trace of a non-trivial $C_3$ irrep over the regular representation vanishes — Serre §2.4, character orthogonality).

This kills the $\text{Im}(h)$ enhancement. The parity-odd channel, which gave the $\sqrt{5}/4$ factor in Class 1 and the $5/3$ factor in Class 2, vanishes at a $C_3$-symmetric vertex. Only the bare $\alpha_1$ survives:

**Coefficient: $1$**

The $\text{Tr}(\sigma_x) = 0$ argument is the character orthogonality theorem applied to the non-trivial $C_3$ representation at a trivalent vertex. This is standard representation theory (Serre §2.4, Theorem 3).

### Step 4: Summary

$$\boxed{\text{Dark correction class} = f(C_3 \text{ quantum number}, \text{parity}, \text{vertex symmetry})}$$

| Class | $C_3$ QN | Parity coupling | Coefficient | Observables |
|-------|---------|----------------|-------------|-------------|
| Amplitude | $\omega^2$ (off-diag) | $\text{Im}[\Sigma]$ | $\sqrt{5}/4 \cdot \alpha_1$ | $V_{us}$, $m_{\nu_2}$, $m_{\nu_3}$ |
| Mass² | $1$ (diagonal) | $\text{Im}^2(h)/\text{Re}^2(h)$ | $(5/3) \cdot \alpha_1$ | $\theta_{23}$ |
| Edge-local | $1$ ($C_3$-sym vertex) | $\text{Tr}(\sigma_x) = 0$ | $1 \cdot \alpha_1$ | $\theta_{13}$, $V_{cb}$ |

The classification is a theorem of $C_3 \times$ parity representation theory. The coefficients are arithmetic from $h = (\sqrt{3} + i\sqrt{5})/2$. No observation is used.

## Result

The dark extraction map is fully determined by the quantum numbers of the observable. This closes `theory_open_items` A.7 (the "linear-vs-squared rule").

## Comparison with experiment

The extraction map itself is not directly observable — it is validated through the downstream predictions that use it. Each downstream parameter file references this map and cites the specific class.

## Open questions

1. **The Higgs VEV coefficient $5/12 = \text{Im}^2(h)/k^*$.** The Higgs VEV correction uses coefficient $\text{Im}^2(h)/k^* = 5/12$, which is $\text{Im}^2(h)$ divided by $k^*$ rather than by $\text{Re}^2(h)$. This is attributed to the VEV being a "quadratic vertex self-energy" (edge-local but with a $k^*$ normalization from the vertex coordination). The derivation from the C₃ structure at the vertex needs explicit computation — it may be a sub-case of Class 3 with partial Im(h) survival.

2. **The photon channel $\sin(\arg h) = \sqrt{5/8}$ for $\beta$.** Cosmic birefringence $\beta$ uses $\text{Im}(h)/|h| = \sin(\arg h)$, not $\text{Im}(h)/|h|^2$. This is attributed to the photon coupling via $\alpha_{\text{EM}}$ rather than $\alpha_1$, with a different Feshbach structure. This is a separate derivation (see `c1_photon_bundle` and `path_c_beta_verify.py`).

3. **Family D per-fermion-leg `c_F` — corrected status (W1, 2026-05-18).** This module's `family_D_per_leg_correction` uses `c_F = -α₁²/(N_atoms·k*) = -α₁²/12`. The result `1/12` is **correct** and is what the framework's real `channel_select` MDL gate (`simulator/gating/mdl.py`) returns when the fermion-leg channel is fixed, *before* enumeration, from the substrate definition (`theorem_car_local_jordan_wigner.md §1`: a Yukawa fermion leg is a single CAR directed-edge mode) — verified in `proofs/foundations/c_F_channel_select_waterfilling_2026-05-18.py` (commit `6c43c54`), with the δ_r gauge-singlet channel reproducing `1/(2|E|)` by the same gate as a consistency anchor.

   **The weakness — well-explained.** The *value* was never the problem; the *expression* was. Family D originally stated `c_F` as "Routes F-1 + F-2, two independent routes, theorem-grade." That is precisely the smuggle `parameter_linter.md` Clause-6c prohibits — an unnamed MDL-bit-cost minimum that conflates `canonical_encoding` with `channel_select` ("BLOCKS Type 6 closure"). F-1 (`1/(N·k*)`) and F-2 (`1/(2|E|)`) are not independent routes; they are **`canonical_encoding`-equivalent** (identical via the Euler identity `2|E|=N·k*`). Because the genuine two-step was never expressed, every hostile audit and the pre-registered single-channel test (prereg #1, commits `788bb45`/`6cd6ccb`) correctly flagged it as broken — and prereg #1's `1/144` was the **gauge-singlet channel's** object applied to the fermion leg, exactly the channel mismatch Clause-6c's `channel_select` step prevents. The corrected, Clause-6-legible two-step is now inlined as `_c_F_denominator_channel_select` here (DAG-contract-compliant: no `proofs/`/`simulator/` import) and written up in `theorem_substrate_feshbach_dark_corrections_master.md §3 (D)`.

   **Residual genuine conditional.** The Step-1 channel fix (single-edge fermion ≠ gauge-singlet democratic sum) is a *structural argument at δ_r's own tier* — the same tier `theorem_unified_oblique.md §6.1` flags for the Perron/h_P form-selection — **not a from-resolvent theorem**. Hence the honest grade is **THEOREM-GRADE-STRUCTURAL, conditional** on that channel argument (per linter Clause 7: named conditional; Clause 8: numeric match unchanged, σ_PDG only; Clause 9a: `c_F ∈ ℚ ⊂ K` — the K-rational substrate analog Clause 9a itself names). It is **not** UNIQUE-THEOREM-GRADE.

## References

- Feshbach, H. (1958). Unified theory of nuclear reactions. *Ann. Phys.* **5**, 357–390.
- Feshbach, H. (1962). Unified theory of nuclear reactions, II. *Ann. Phys.* **19**, 287–313.
- Sakurai, J.J. (2020). *Modern Quantum Mechanics*, 3rd ed. Cambridge University Press. §5.2 (degenerate perturbation theory).
- Serre, J.-P. (1977). *Linear Representations of Finite Groups*. Springer GTM 42. §2.4 (character orthogonality).
