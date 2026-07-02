# Derivation: Local emergent Lorentzian metric signature at the Γ Dirac cone of srs

## Abstract

We derive the local emergent metric tensor of the substrate scalar Bloch
Hamiltonian $H(\mathbf k)$ on srs at the $\Gamma$ Dirac cone. The lower three
bands of $H(\mathbf k)$ are triply-degenerate at $\lambda_* = -1$ at $\Gamma$
(the $K_4$ adjacency spectrum, Biggs 1993 §2.2) and split linearly off
$\Gamma$ as a **spin-1 Dirac cone** with Cartesian-isotropic Fermi velocity

$$v_F^\Gamma \;=\; \tfrac12 \qquad\text{(lattice-constant per substrate-tick units)}$$

(`predictions/srs_dirac_cone_velocities.py`, theorem-grade). The two
dispersing bands satisfy the relativistic mass-shell

$$ (E - \lambda_*)^2 \;=\; v_F^2\,|\mathbf k_{\rm cart}|^2, $$

from which the local emergent metric reads off as
$\eta^{\mu\nu} = \mathrm{diag}(-1, v_F^2, v_F^2, v_F^2)$ in lattice-constant
units, equivalent to standard Minkowski

$$ \boxed{\;\eta_{\mu\nu} \;=\; \mathrm{diag}(-1,\, +1,\, +1,\, +1)\;}$$

after the local time rescaling $\tau = v_F\,t$. The signature
$(n_-, n_+) = (1, 3)$ is independent of the rescaling. This is the
**leading-order, locally-emergent Lorentzian signature** of the substrate
spacetime; lifting the local Minkowski cone to a globally-emergent Lorentzian
manifold is research-level (Iorio-elastic vielbein + Einstein backreaction;
internal working notes).

This theorem **establishes the wave-engine `LORENTZ_SIG` tag locally**
(operator 6.10, "Lorentzian metric") and corrects the framework's prior
posture in which $(-,+,+,+)$ signature was a load-bearing premise of
`docs/theorems/theorem_lorentz_causal_sector.md` (line 62). The premise is now
derived at theorem grade in the local-cone neighbourhood.

## Framework axioms invoked

- **A1** (binary self-inverse toggle): the substrate is the discrete walker
  on srs; $H(\mathbf k)$ is the scalar Bloch fibre on the four-atom Wyckoff 8a
  primitive cell (vertex level).
- **A3** (complex Hilbert): $H(\mathbf k) \in \mathbb C^{4\times 4}$ Hermitian.

## Cited mathematical theorems

- **Biggs 1993**, *Algebraic Graph Theory* 2nd ed., Cambridge, §2.2:
  $\operatorname{spec}(A_{K_n}) = \{n - 1, -1, \ldots, -1\}$.
- **Kato 1980**, *Perturbation Theory for Linear Operators* 2nd ed.,
  Springer Grundlehren **132**, §II.5 Theorem 5.11 (degenerate
  Rayleigh-Schrödinger perturbation): for an isolated $m$-fold degenerate
  eigenvalue $\lambda_0$ of $H_0$ with eigenprojector $P_0$, the leading
  correction under $H_0 + V$ is given by the eigenvalues of $P_0 V P_0$
  acting on $\mathrm{ran}(P_0)$.
- **Wigner-Eckart** (Hamermesh 1962, *Group Theory and its Application to
  Physical Problems*, Theorem 9.5; Inui-Tanabe-Onodera 1990, *Group Theory
  and its Applications in Physics*, Ch. 7): for irreps $\Gamma_{\rm vec}$
  of vectors and $\Gamma_{\rm st}$ of states, matrix elements
  $\langle \Gamma_{\rm st}, \alpha | \mathbf O_i | \Gamma_{\rm st}, \beta\rangle$
  factorise into a single reduced matrix element times Clebsch-Gordan
  coefficients.
- **Sakurai (1994), *Modern Quantum Mechanics*, §3.5** / **Edmonds 1957,
  *Angular Momentum in Quantum Mechanics*, §2**: the Hermitian spin-1
  generators on the $j = 1$ representation of $\mathrm{SO}(3)$ satisfy
  $[S_a, S_b] = i\,\epsilon_{abc}\,S_c$ and have Casimir
  $S_x^2 + S_y^2 + S_z^2 = 2 \cdot \mathbf 1$. Eigenvalues of
  $\mathbf k \cdot \mathbf S$ along any direction $\hat{\mathbf k}$ are
  $\{+|\mathbf k|, 0, -|\mathbf k|\}$.

## Derivation

### Step 1 — Spin-1 Dirac form at the Γ cone

By the upstream theorem `predictions/srs_dirac_cone_velocities.py` (Biggs +
Kato §II.5 + Wigner-Eckart on the cubic-432 $T$-irrep, sympy-verified), the
effective Hamiltonian on the 3-dimensional $\lambda_* = -1$ eigenspace at
$\Gamma$ takes the closed-form

$$ H_{\rm eff}(\mathbf k_{\rm cart}) \;=\; \lambda_*\,\mathbf 1_3 \;+\; v_F\,\bigl(k_x\,S_x + k_y\,S_y + k_z\,S_z\bigr) $$

with $\lambda_* = -1$, $v_F = \tfrac12$, and $(S_x, S_y, S_z)$ the Hermitian
spin-1 Cartesian generators on the $T$-irrep. The Wigner-Eckart theorem forces
this proportionality to a **single reduced matrix element** times the
Clebsch-Gordan structure; the closed-form $v_F = 1/2$ is the explicit value of
that matrix element.

`proofs/foundations/lorentz_sig_spin1_dirac_decomposition.py` verifies, in
sympy exact arithmetic, that the generators close to the **full** SO(3)
algebra $[S_a, S_b] = i\,\epsilon_{abc}\,S_c$, not merely the cubic-432 vector
representation. This SO(3) closure is the **leading-order emergent rotational
invariance**; sub-leading orders ($k^3, k^4, \ldots$) carry only cubic-432
symmetry and source the dim-6 LV anisotropy
$\eta^H_{\rm NB} = 1/6$ (`predictions/srs_bloch_lv_dim6.py`).

### Step 2 — Eigenvalues of $\mathbf k \cdot \mathbf S$

The characteristic polynomial of $M(\mathbf k) := k_x S_x + k_y S_y + k_z S_z$
on the $j = 1$ representation factors as

$$ \det\bigl(M(\mathbf k) - \lambda\,\mathbf 1_3\bigr) \;=\; -\lambda\,\bigl(\lambda^2 - |\mathbf k|^2\bigr). $$

This is verified symbolically by `predictions/lorentz_signature_local.py:_kdotS_eigenvalues_factorise`
on the explicit Cartesian generators

$$
S_x = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & -i \\ 0 & i & 0 \end{pmatrix},\qquad
S_y = \begin{pmatrix} 0 & 0 & i \\ 0 & 0 & 0 \\ -i & 0 & 0 \end{pmatrix},\qquad
S_z = \begin{pmatrix} 0 & -i & 0 \\ i & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix},
$$

with $|\mathbf k|^2 = k_x^2 + k_y^2 + k_z^2$.

The three eigenvalues of $H_{\rm eff}$ are therefore

$$ E_+(\mathbf k) = \lambda_* + v_F |\mathbf k|,\qquad E_0(\mathbf k) = \lambda_*,\qquad E_-(\mathbf k) = \lambda_* - v_F |\mathbf k|. $$

The two bands $E_\pm$ are **linearly dispersing** with isotropic slope
$\pm v_F$. The flat band $E_0 = \lambda_*$ is the **longitudinal/zero-mode**,
analogous to the longitudinal photon polarisation that becomes pure-gauge
after fixing.

### Step 3 — Mass-shell relation

Squaring the dispersing-band relation and rearranging:

$$ (E - \lambda_*)^2 \;=\; v_F^2\,|\mathbf k_{\rm cart}|^2 \quad\Leftrightarrow\quad -(E - \lambda_*)^2 \;+\; v_F^2\bigl(k_x^2 + k_y^2 + k_z^2\bigr) \;=\; 0. $$

This is the **massless relativistic mass-shell** in the form
$\eta^{\mu\nu} p_\mu p_\nu = 0$ with $p_0 = E - \lambda_*$, $p_i = k_{\rm cart}^i$,
and the inverse metric

$$ \eta^{\mu\nu} \;=\; \mathrm{diag}(-1,\, v_F^2,\, v_F^2,\, v_F^2). $$

By inversion, the metric tensor in lattice-constant units is

$$ \boxed{\;\eta_{\mu\nu} \;=\; \mathrm{diag}\bigl(-1,\, 1/v_F^2,\, 1/v_F^2,\, 1/v_F^2\bigr) \;=\; \mathrm{diag}(-1,\, 4,\, 4,\, 4)\quad\text{at}\;v_F = 1/2.\;}$$

(Some authors write the metric "with $v_F$ on the spatial entries"; the choice
between $\eta^{\mu\nu}$ and $\eta_{\mu\nu}$ is a convention. The physical content
— the lightcone — is the same: lightlike vectors satisfy
$|\mathbf k_{\rm cart}| = (E - \lambda_*)/v_F$.)

### Step 4 — Time rescaling τ = v_F t

The local rescaling $\tau = v_F\,t$ (i.e., measure time in units of
$1/v_F$ substrate-ticks per lattice-constant of light travel) eliminates the
$v_F$ factors. With $E_\tau := E / v_F$:

$$ -E_\tau^2 \;+\; |\mathbf k_{\rm cart}|^2 \;=\; 0,\qquad \eta_{\mu\nu} \;=\; \mathrm{diag}(-1,\, +1,\, +1,\, +1). $$

This is the **standard Minkowski metric** with signature $(-, +, +, +)$. The
substrate's emergent speed of light is $c = v_F$ in lattice-constant per
substrate-tick units; equivalently, after the $\tau$ rescaling, $c = 1$.

### Step 5 — Signature (1, 3) is rescaling-invariant

For any $v_F > 0$, the eigenvalues of $\eta_{\mu\nu} = \mathrm{diag}(-1, v_F^2, v_F^2, v_F^2)$
are one negative ($-1$) and three positive ($v_F^2$ each). Therefore

$$ \boxed{\;(n_-, n_+) \;=\; (1, 3),\;}$$

i.e., **Lorentzian signature**, independently of the time-rescaling
convention. Euclidean $(0, 4)$ and split $(2, 2)$ signatures are excluded.

This concludes the local theorem.

## Result

$$ \boxed{\quad \eta_{\mu\nu}\big|_{\Gamma\text{-cone}} \;=\; \mathrm{diag}(-1,\, +1,\, +1,\, +1),\qquad (n_-, n_+) = (1, 3).\quad} $$

The local emergent metric at the $\Gamma$ Dirac cone of srs is the standard
Minkowski metric of $(3+1)$-dimensional special relativity, with one time-like
and three space-like directions. The emergent local speed of light is
$c = v_F^\Gamma = 1/2$ in lattice-constant per substrate-tick units.

## Comparison with experiment

The metric signature of physical spacetime is **observed Lorentzian
$(-, +, +, +)$ in any local inertial frame** — established operationally by
the constancy of $c$ (Michelson-Morley 1887, Kennedy-Thorndike 1932), the
lightcone structure of every scattering and decay process since Compton 1923,
weak-field GR tests (Eddington 1919, Pound-Rebka 1959), and gravitational-wave
detection (LIGO 2015). Standard references: Wald 1984 §1.1, Misner-Thorne-Wheeler
1973 §1.1, Weinberg 1972 §2.1.

The substrate prediction $(n_-, n_+) = (1, 3)$ matches the observed signature
**exactly** (structural match; no numerical fit). The numerical value of
$v_F^\Gamma = 1/2$ is in lattice-constant per substrate-tick units; converting
to SI units requires the framework's lattice-constant $\to$ metres and
substrate-tick $\to$ seconds conversions, which are set elsewhere in the
framework (Higgs VEV, Planck mass, substrate update rate). The signature
itself is convention-independent.

## Open questions

1. **Global lift to a Lorentzian manifold.** The local Minkowski cone at the
   $\Gamma$-cone is theorem-grade above. Lifting to a global Lorentzian
   manifold under slow elastic deformations of srs (Iorio-elastic regime) is
   scoped at theorem-grade-pending in
   an internal working note. The vielbein
   $e^a{}_b = \delta^a_b + \partial_b u^a$ has been derived ($\beta = 1$,
   `proofs/foundations/lorentz_sig_iorio_session2_*.py`) and the spin connection
   $\omega = \tfrac14\,\Omega\cdot(\mathbf k \times \mathbf S)$
   (`lorentz_sig_iorio_session3_spin_connection.py`). The remaining gap is the
   continuum-limit theorem promoting "local Minkowski at every $\mathbf x$" to
   "global Lorentzian manifold." Estimated 3–5 sessions.
2. **Backreaction / emergent Newton constant $G_{\rm sub}$.** The linearised
   Einstein equation $-\Box u^{ab} = 8\pi G_{\rm sub}\,T^{ab}$ is structural
   (`proofs/foundations/lorentz_sig_iorio_session4_einstein.py`,
   an internal working note); the dimensionless
   coefficient $G_{\rm sub}$ requires the substrate's elastic moduli — a
   pure-graph-theory computation pending future sessions.
3. **Multi-valley resolution.** srs has additional Dirac cones at $H$
   (particle-hole conjugate of $\Gamma$, same $v_F = 1/2$) and at $P$
   ($v_F = \sqrt 3 / 6$). MDL ranking selects the $\Gamma + H$ pair as the
   dominant emergent Dirac sector (a 2-valley structure analogous to graphene's
   $K, K'$); $P$ cones are sub-leading. Details in
   an internal working note. Sub-dominant
   cones are physically relevant to the **global** signature lift, not the
   local result derived here.
4. **Lorentzian-NCG / Krein-space alternatives.** Routes C-i (Besnard-Bizi-Iochum
   Lorentzian spectral triples) and C-ii (BLMS causal-set continuum limit) remain
   open as alternative routes to the global lift; they are de-prioritized given
   the local result above.

## References

### Cited theorems
- Biggs, N. (1993). *Algebraic Graph Theory*, 2nd ed. Cambridge Univ. Press.
- Kato, T. (1980). *Perturbation Theory for Linear Operators*, 2nd ed.,
  Springer Grundlehren **132**.
- Hamermesh, M. (1962). *Group Theory and Its Application to Physical Problems*.
  Addison-Wesley.
- Inui, T., Tanabe, Y., Onodera, Y. (1990). *Group Theory and Its Applications
  in Physics*. Springer.
- Sakurai, J. J. (1994). *Modern Quantum Mechanics*, 2nd ed. Addison-Wesley.
- Edmonds, A. R. (1957). *Angular Momentum in Quantum Mechanics*. Princeton.

### Framework upstream
- `predictions/srs_dirac_cone_velocities.py` + `_derivation.md` — closed-form
  $v_F^\Gamma = 1/2$ via Kato §II.5 + Wigner-Eckart on the $T$-irrep
  (theorem-grade, sympy-verified).
- `predictions/k_star.py`, `predictions/d_spatial.py`,
  `predictions/g_girth_derivation.md` — substrate identification at theorem grade.
- `proofs/foundations/lorentz_sig_spin1_dirac_decomposition.py` — explicit SO(3)
  closure of the substrate's spin-1 generators, Casimir verification, and
  derivation of the emergent rotational invariance at leading order.
- `proofs/foundations/lorentz_sig_dirac_cone_symbolic.py` — full sympy
  diagonalisation of the 4×4 Bloch H at $\Gamma, H, P, N$.
- `predictions/srs_bloch_lv_dim6.py` + `_derivation.md` — sub-leading dim-6
  LV anisotropy ($\eta^H_{\rm NB} = 1/6$), the leading correction to local
  SO(3) invariance.
- `docs/theorems/theorem_lorentz_causal_sector.md` — Stage 3 leading-order Lorentz
  invariance theorem; line 62 previously stated "(3+1) signature is assumed"
  and is now replaced by the local derivation here.

### Open-problem scoping
  identifying Routes A/B/C; updated 2026-04-26 with Route C bounded-$D^2$
  obstruction.
  resolution and global-lift research path.
  scoping (vielbein + spin connection + curved-space spin-1 Dirac).
  backreaction scoping (linearised Einstein eq. + $G_{\rm sub}$).
  (de-prioritized given the local theorem here).
