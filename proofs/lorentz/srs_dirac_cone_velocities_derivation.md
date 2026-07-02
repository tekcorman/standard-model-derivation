# Derivation: srs scalar-Bloch Dirac-cone Fermi velocities

**Audit anchor:** Foundational Lorentz-arc theorem. Conditional on Rows 4, 6 of `docs/audits/registers/uniqueness_ledger.md` (k* = 3 + srs identification). Establishes the wave-engine LORENTZ_SIG tag locally via op 6.10 per `docs/theorems/lorentz_sig_ccclose_joint_closure.md`.

## Abstract

The 4-band scalar Bloch Hamiltonian $H(\mathbf k)$ on the srs primitive cell
(I4₁32 + Wyckoff 8a + nearest-neighbour bonds) has Hermitian spectrum
$\{3, -1, -1, -1\}$ at $\Gamma$. The 3-fold degeneracy at $\lambda = -1$ splits
linearly off $\Gamma$ as a **spin-1 Dirac cone** with
Cartesian-isotropic Fermi velocity $v_F^\Gamma = 1/2$ (in lattice-constant
per substrate-tick units). At $P = (\tfrac14,\tfrac14,\tfrac14)_{\rm frac}$ the
spectrum is $\{\pm\sqrt 3\}$ (each with multiplicity 2); each 2-fold cluster
splits as a **2-band Dirac cone** with Cartesian-isotropic Fermi velocity
$v_F^P = \sqrt 3/6 = 1/(2\sqrt 3)$. The $H = (-\tfrac12,\tfrac12,\tfrac12)_{\rm frac}$
spectrum $\{-3, 1, 1, 1\}$ inherits the $\Gamma$ Dirac-cone structure by
particle-hole conjugation. At $N$ the spectrum has all simple eigenvalues
$\{\pm\sqrt 5, \pm 1\}$ — no Dirac cone. All results are derived from
$\mathbf{A_1}$ (toggle), $\mathbf{A_3}$ (complex Hilbert), and the upstream
theorem $g$-girth $\Rightarrow$ srs (Sunada 2012); the only cited external
fact is the elementary spectrum of $K_4$ adjacency (Biggs 1993 §2.2).

These v_F values are the **structural-emergent** Fermi velocities of the
substrate's Dirac cones, before any MDL-cone selection or vacuum
identification. The MDL ranking that selects $\Gamma$ as the dominant cone
lives in an internal working note and is
not load-bearing for this theorem.

## Framework axioms invoked

- **A1** (binary self-inverse toggle): the substrate is the discrete walker
  on srs; $H(\mathbf k)$ is the scalar Bloch fibre on the four-atom primitive
  cell (vertex-level operator).
- **A3** (complex Hilbert): $H(\mathbf k) \in \mathbb C^{4\times 4}$ Hermitian.

The upstream identification of srs as the substrate (`predictions/k_star.py`,
`predictions/d_spatial.py`, `predictions/g_girth_derivation.md` §2 invoking
Sunada 2012) is theorem-grade; we use only its specific bond list, which is
gauge-equivalent to the cell_edges of `proofs/foundations/theorem_B2_signature.py`
and to `proofs/common.find_bonds()`.

## Cited mathematical theorems

- **Biggs 1993**, *Algebraic Graph Theory* 2nd ed., Cambridge, §2.2.
  $\operatorname{spec}(A_{K_n}) = \{n-1, -1, -1, \ldots, -1\}$.
- **Kato 1980**, *Perturbation Theory for Linear Operators* 2nd ed.,
  Springer Grundlehren **132**, §II.5 Theorem 5.11 (degenerate Rayleigh–Schrödinger
  perturbation): for an isolated $m$-fold degenerate eigenvalue $\lambda_0$ of
  $H_0$ with eigenprojector $P_0$, the leading correction under
  $H_0 + V$ is given by the eigenvalues of $P_0 V P_0$ acting on
  $\operatorname{ran}(P_0)$.
- **Wigner–Eckart** (e.g. Hamermesh 1962 *Group Theory*, Thm. 9.5;
  Inui–Tanabe–Onodera 1990 *Group Theory and its Applications in Physics*,
  Ch. 7): for an irrep $\Gamma_{\rm vec}$ of vectors and an irrep $\Gamma_{\rm st}$
  of states, matrix elements $\langle \Gamma_{\rm st}, \alpha | \mathbf O_i | \Gamma_{\rm st}, \beta\rangle$
  factorise into a single reduced matrix element times a Clebsch–Gordan
  coefficient.

## Derivation

### Step 1 — Bloch Hamiltonian construction (A1, A3)

The srs primitive cell has 4 atoms at Wyckoff 8a positions, with each atom
having 3 NN to neighbouring cells. The 6 undirected NN edges in the symbolic
gauge of `proofs/foundations/theorem_B2_signature.py` are

$$
\bigl\{(0,1,(1,1,1)),\, (0,2,(1,1,1)),\, (0,3,(1,1,1)),\, (1,2,(-1,0,0)),\, (1,3,(0,1,0)),\, (2,3,(0,0,-1))\bigr\}
$$

producing 12 directed bonds upon adding reverses. The Hermitian Bloch operator is

$$
H_{ij}(\mathbf k) \;=\; \sum_{\text{bonds } (i \leftarrow j,\,\mathbf n)} \exp\!\bigl(2\pi i\, \mathbf k \cdot \mathbf n\bigr).
$$

This is the scalar (vertex-level) Bloch fibre on the 4-atom unit cell. By
construction $H(\mathbf k)^\dagger = H(\mathbf k)$.

### Step 2 — Spectrum at $\Gamma$ (Biggs)

Setting $\mathbf k = 0$, all phases collapse to $1$ and $H(0) = J - I$ where
$J$ is the $4\times 4$ all-ones matrix. This is the adjacency matrix of $K_4$.
By Biggs 1993 §2.2:

$$
\operatorname{spec}\bigl(A_{K_4}\bigr) \;=\; \{3,\, -1,\, -1,\, -1\}.
$$

The single eigenvalue $+3$ has eigenvector $\mathbf v_0 = (1,1,1,1)/2$ (the
Perron eigenstate; matches the framework's vacuum identification). The 3-fold
degenerate eigenvalue $-1$ has eigenspace $\mathbf v_0^\perp$, a 3-dimensional
subspace.

### Step 3 — Spectra at $H$, $P$, $N$ (sympy diagonalisation)

The Bloch matrices at the other three high-symmetry sites are direct
substitutions:

$$
\begin{aligned}
H(H) &= \mathrm{diag}\text{-zero matrix with all off-diagonal entries }= -1, \\
H(P) &= \begin{pmatrix} 0 & i & i & i \\ -i & 0 & i & -i \\ -i & -i & 0 & i \\ -i & i & -i & 0 \end{pmatrix}, \\
H(N) &= \text{matrix with entries}\ \pm 1\ \text{and zero diagonal.}
\end{aligned}
$$

sympy's exact diagonalisation gives:

$$
\operatorname{spec} H(H) = \{-3,\, +1,\, +1,\, +1\},\qquad \operatorname{spec} H(P) = \{+\sqrt 3,\, +\sqrt 3,\, -\sqrt 3,\, -\sqrt 3\},
$$
$$
\operatorname{spec} H(N) = \{+\sqrt 5,\, +1,\, -1,\, -\sqrt 5\}.
$$

The $H$-spectrum is the additive negative of the $\Gamma$-spectrum (a
particle-hole conjugation specific to this Bloch gauge). The $P$-spectrum
has two 2-fold clusters at $\pm\sqrt 3$. The $N$-spectrum has all simple
eigenvalues — no Dirac candidate.

### Step 4 — Linear perturbation $V_1(\mathbf k)$ at $\Gamma$ (Kato §II.5)

Expanding the phase $\exp(2\pi i\, \mathbf k \cdot \mathbf n) = 1 + 2\pi i\,(\mathbf k \cdot \mathbf n) + O(\mathbf k^2)$
and subtracting $H(0)$, the linear-in-$\mathbf k$ part of $H$ is

$$
[V_1]_{i j}(\mathbf k) \;=\; \sum_{\text{bonds }(j \to i,\, \mathbf n)} 2\pi i\, (\mathbf k \cdot \mathbf n).
$$

By construction $V_1$ is Hermitian and traceless (each undirected edge
$(j,i,\mathbf n)$ contributes $2\pi i\,(\mathbf k \cdot \mathbf n)$ to $[V_1]_{ij}$ and
$-2\pi i\,(\mathbf k \cdot \mathbf n)$ to $[V_1]_{ji}$, summing to zero on the diagonal).

For the Perron state $\mathbf v_0$:

$$
\langle \mathbf v_0 | V_1 | \mathbf v_0\rangle \;=\; 0,
$$

confirming that the first-order correction to the $\lambda = +3$ band
vanishes — consistent with the **quadratic** dispersion of the top band
$\lambda_0(\mathbf q) = 3 - |\mathbf q|^2/16$ proved in
`predictions/srs_bloch_dispersion_gamma.py`.

### Step 5 — Projection onto the $\lambda = -1$ subspace (Kato §II.5 Thm 5.11)

Pick the orthonormal basis

$$
\mathbf g_1 = \tfrac{1}{\sqrt 2}(1,-1,0,0)^\top,\quad
\mathbf g_2 = \tfrac{1}{\sqrt 6}(1,1,-2,0)^\top,\quad
\mathbf g_3 = \tfrac{1}{\sqrt{12}}(1,1,1,-3)^\top.
$$

These span $\mathbf v_0^\perp$ and are eigenvectors of $H(0) = J - I$ at
eigenvalue $-1$. The $3\times 3$ projection $M(\mathbf k) = G^\dagger V_1(\mathbf k) G$,
where $G = (\mathbf g_1\mid\mathbf g_2\mid\mathbf g_3)$, is computed by sympy:

$$
M(\mathbf k) \;=\; \frac{i\,\pi}{6}\!\begin{pmatrix} 0 & 4\sqrt 3\, k_1 & \sqrt 6\,(-k_1 - 3 k_2) \\
-4\sqrt 3\, k_1 & 0 & 3\sqrt 2\,(k_1 + k_2 + 2 k_3) \\
\sqrt 6\,(k_1 + 3 k_2) & -3\sqrt 2\,(k_1 + k_2 + 2 k_3) & 0
\end{pmatrix}.
$$

By Kato §II.5 Thm 5.11, the eigenvalues of $M(\mathbf k)$ are the leading-order
band corrections $\delta\lambda$ to $\lambda = -1$.

### Step 6 — Eigenvalues of $M(\mathbf k)$: spin-1 Dirac structure

Direct symbolic computation gives:

$$
\operatorname{tr} M = 0,\qquad \det M = 0,\qquad \operatorname{tr}(M^2) \;=\; 4\pi^2 \cdot Q(\mathbf k)
$$

where $Q(\mathbf k) = k_1^2 + k_1 k_2 + k_1 k_3 + k_2^2 + k_2 k_3 + k_3^2$.
Since $\operatorname{tr} M = \det M = 0$, the characteristic polynomial of
$M$ is $\mu^3 - \tfrac12 \operatorname{tr}(M^2) \mu = 0$, hence

$$
\operatorname{spec} M(\mathbf k) \;=\; \bigl\{+a(\mathbf k),\, 0,\, -a(\mathbf k)\bigr\},\qquad a(\mathbf k)^2 = \tfrac12 \operatorname{tr}(M^2) = 2\pi^2\, Q(\mathbf k).
$$

This is the **spin-1 Dirac structure**: one flat band at $\lambda = -1$ and
two linearly-dispersing bands $\lambda = -1 \pm a(\mathbf k)$.

### Step 7 — Cartesian isotropy at $\Gamma$ (Wigner–Eckart corollary)

Cartesian momentum at $\Gamma$ is

$$
\mathbf k_{\rm cart} \;=\; k_1\,\mathbf b_1 + k_2\,\mathbf b_2 + k_3\,\mathbf b_3,\qquad
\mathbf b_i = 2\pi\,(\mathbf{\hat e}_{i+1} + \mathbf{\hat e}_{i+2})\ \text{(BCC reciprocal)},
$$

giving $|\mathbf k_{\rm cart}|^2 = 8\pi^2\, Q(\mathbf k)$. Therefore

$$
\boxed{\;\frac{a(\mathbf k)^2}{|\mathbf k_{\rm cart}|^2} \;=\; \frac{2\pi^2 Q}{8\pi^2 Q} \;=\; \tfrac14,\qquad a(\mathbf k) = \tfrac12\, |\mathbf k_{\rm cart}|.\;}
$$

The ratio is direction-independent, hence the Dirac cone at $\Gamma$ is
**isotropic in Cartesian k-space** with Fermi velocity

$$
\boxed{\;v_F^\Gamma \;=\; \tfrac12\quad\text{(lattice-constant per substrate-tick).}\;}
$$

The structural-symmetry reading: the 3-d $\lambda = -1$ subspace at $\Gamma$
carries the unique 3-d irrep of the cubic point group 432 (the standard
$T$-vector irrep). $V_1(\mathbf k)$ is a vector operator transforming as the
same irrep. By the Wigner–Eckart theorem, all matrix elements
$\langle T;\alpha\,|\,V_{1,i}\,|\,T;\beta\rangle$ are proportional to a single
reduced matrix element times Clebsch–Gordan coefficients — and hence
$M(\mathbf k) = v_F\,(\mathbf k_{\rm cart}\cdot \mathbf S)$ for the spin-1
generators $\mathbf S$ on the $T$-irrep, whose eigenvalues along any unit
direction $\hat{\mathbf k}$ are $\{+1, 0, -1\}$. The closed form
$v_F = 1/2$ is the explicit value of the reduced matrix element. The
sympy computation of Step 6 is the explicit instance of this Wigner–Eckart
theorem.

### Step 8 — Same construction at $P$

At $P = (\tfrac14,\tfrac14,\tfrac14)$, the Bloch matrix has eigenvalue
$-\sqrt 3$ with 2-dim eigenspace. Pick the orthonormal sympy-computed basis
$\{\mathbf u_1, \mathbf u_2\}$ of this 2-dim subspace. The
linear-in-$\delta\mathbf k$ perturbation $V_1^{(P)}(\delta\mathbf k)$ at $P$
shares the bond structure of $V_1$ at $\Gamma$ but with phase factors
$\exp(2\pi i\, P \cdot \mathbf n)$ in front of each bond's
$2\pi i\,(\delta\mathbf k \cdot \mathbf n)$. Direct sympy computation of
the $2\times 2$ projection $M_P(\delta\mathbf k) = U^\dagger V_1^{(P)} U$ gives:

$$
\operatorname{tr} M_P = 0,\qquad -\det M_P \;=\; \tfrac{2\pi^2}{3}\, Q(\delta\mathbf k).
$$

For a 2-d Hermitian traceless matrix, $-\det = a^2$ where $\pm a$ are the
eigenvalues. Hence

$$
a^P(\delta\mathbf k)^2 \;=\; \tfrac{2\pi^2}{3}\, Q(\delta\mathbf k) \;=\; \tfrac{1}{12}\,|\delta\mathbf k_{\rm cart}|^2,\qquad
\boxed{\;v_F^P \;=\; \frac{\sqrt 3}{6} \;=\; \frac{1}{2\sqrt 3}.\;}
$$

The $\det M_P = 0$ check is unnecessary (the 2-dim cluster has eigenvalues
$\pm a$ irrespective). The same construction at the upper $+\sqrt 3$ cluster
gives the identical $v_F^P$ by complex conjugation symmetry of $H(P)$.

### Step 9 — Particle-hole at $H$

The bond list and gauge choice produce $H(H = (-\tfrac12,\tfrac12,\tfrac12))$
that is the entrywise negative of $H(\Gamma)$. Hence
$\operatorname{spec} H(H) = -\operatorname{spec} H(\Gamma) = \{-3, 1, 1, 1\}$,
and the 3-fold cluster at $\lambda = +1$ inherits the spin-1 Dirac structure
with the **same** $v_F^H = 1/2$.

## Result

$$
\boxed{\;v_F^\Gamma \;=\; \tfrac12,\qquad v_F^H \;=\; \tfrac12,\qquad v_F^P \;=\; \tfrac{\sqrt 3}{6}.\;}
$$

All three are exact rationals (or exact radicals) in lattice-constant
per substrate-tick units. Each cone is **Cartesian-isotropic**, hence each
delivers a **local emergent (1+3) Minkowski cone** with effective light-speed
equal to the cone's $v_F$.

Numerical cross-check (`proofs/foundations/lorentz_sig_dirac_cone_refined.py`)
gives $\operatorname{spread}(\varepsilon)/|\mathbf k_{\rm cart}|$ converging
to $1.0000000$ at $\Gamma/H$ and $0.5773503$ at $P$ across seven directions
each, at relative precision $10^{-7}$ between successive $\varepsilon$
decades. These match the symbolic theorems $2 v_F^\Gamma = 1$ and
$2 v_F^P = 1/\sqrt 3 \approx 0.5773503$ to all displayed digits.

## Comparison with experiment

N/A — foundational structural theorem about the substrate spectral
operator's Dirac-cone fine structure. No direct numerical observable
because the conversion of "lattice constant per substrate tick" to SI
units is set by other parts of the framework (Higgs VEV, Planck mass,
substrate update rate). The downstream physical impact is the **local
emergent Lorentzian signature** at each Dirac cone — see
an internal working note.

## Open questions

1. **Multi-valley resolution.** The framework has at least three
   structurally distinct Dirac cones (Γ-triple, H-triple, P-double).
   Their Fermi velocities are not equal ($1/2$ vs $\sqrt 3/6$), so they do
   not share a single emergent speed of light. The MDL ranking in
   an internal working note selects $\Gamma$
   as dominant on orbit-size + Bloch-state grounds, but the formal MDL
   bit-cost prescription used to break the $\Gamma$-vs-$H$ tie is heuristic.
   Research-level (~5–10 sessions).
2. **Match to existing LV EFT coefficients.** The framework's existing
   $\eta_5 = 0$ (`predictions/eta_5_lorentz_dim5.py`) and $\eta_{\rm NB} = 1/12$
   (`predictions/eta_lattice_lorentz_dim6.py`) might be predictable from
   the second-order multi-valley contributions of the sub-dominant cones at
   $H$ and $P$. Conjectural pending explicit second-order Kato. Research-level
   (~3–5 sessions).
3. **Local-to-global signature lift.** The Wigner–Eckart-style isotropy
   gives a local Minkowski cone at each Dirac point. Lifting to a global
   Lorentzian manifold requires either a Sorkin-style continuum limit or
   a Connes-Krein NCG construction. Research-level (~10+ sessions);
   complementary to Routes C-i and C-ii in an internal note.

## References

- Biggs, N. (1993). *Algebraic Graph Theory*, 2nd ed. Cambridge Univ. Press.
- Kato, T. (1980). *Perturbation Theory for Linear Operators*, 2nd ed.
  Springer Grundlehren **132**.
- Hamermesh, M. (1962). *Group Theory and Its Application to Physical Problems*.
  Addison-Wesley.
- Inui, T., Tanabe, Y., Onodera, Y. (1990). *Group Theory and Its Applications
  in Physics*. Springer.
- Sunada, T. (2012). *Topological Crystallography: With a View Towards Discrete
  Geometric Analysis*. Springer.
- `predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth_derivation.md`
  — substrate identification at theorem grade.
- `predictions/srs_bloch_dispersion_gamma.py` — quadratic dispersion of the top
  (Perron) band at $\Gamma$.
- `predictions/B_P_doubly_degenerate_h.py` — sibling $P$-point Bloch theorem
  on the Hashimoto operator (12-band non-Hermitian; orthogonal to the scalar-Bloch
  theorems here).
- `predictions/theorem_B2_signature.py` — $P$-point Ramanujan projector form
  giving Cl(6,0) substrate signature; the bond-list gauge here matches.
- `proofs/foundations/lorentz_sig_dirac_cone_symbolic.py` — full symbolic
  verification script (this derivation is its journal write-up).
- `proofs/foundations/lorentz_sig_dirac_cone_refined.py` — numerical
  cross-check at seven directions per site.
- `proofs/foundations/lorentz_sig_orbit_sizes.py` — cubic-432 orbit-size
  enumeration for the MDL scoping doc.
  and multi-valley research-level open questions.
