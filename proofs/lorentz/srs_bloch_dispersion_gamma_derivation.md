# Derivation of `srs_bloch_dispersion_gamma`

## Abstract

We derive the closed-form coefficients of the small-$\boldsymbol{k}$
expansion of the Perron eigenvalue $\lambda_0(\boldsymbol{k})$ of the
srs scalar Bloch adjacency $A(\boldsymbol{k})$ near the Brillouin-zone
origin $\Gamma$. With the same I4$_132$ + Wyckoff 8a primitive-cell
bond list used in `predictions/B_P_doubly_degenerate_h.py`, the second-
order non-degenerate Rayleigh–Schrödinger correction (Kato 1980 §II.5
Theorem 5.4) gives

$$\lambda_0(\boldsymbol{k}) \;=\; k^{*} \;-\; \sum_{a, b = 1}^{3} \gamma_{ab}\, k_a\,k_b \;+\; O(|\boldsymbol{k}|^{4}),$$

where $(k_1, k_2, k_3)$ are reduced coordinates conjugate to the
**primitive** BCC lattice basis and the symmetric tensor

$$\gamma_{ab} \;=\; \frac{\pi^{2}}{2}\,\delta_{ab} \;+\; \frac{\pi^{2}}{4}\,(1 - \delta_{ab})$$

is independent of $\boldsymbol{k}$. Equivalently, in the physical
Cartesian wavevector $\boldsymbol{q} = 2\pi\,(k_1\,\boldsymbol{b}_1 + k_2\,\boldsymbol{b}_2 + k_3\,\boldsymbol{b}_3)$
with primitive BCC reciprocal vectors $\boldsymbol{b}_1 = (0,1,1)$,
$\boldsymbol{b}_2 = (1,0,1)$, $\boldsymbol{b}_3 = (1,1,0)$ (and
conventional cubic lattice constant $a = 1$),

$$\boxed{\;\lambda_0(\boldsymbol{q}) \;=\; k^{*} \;-\; \frac{|\boldsymbol{q}|^{2}}{16} \;+\; O(|\boldsymbol{q}|^{4})\;}$$

i.e. the dispersion is **isotropic in physical Cartesian space** with
the closed scalar coefficient $\gamma_{\mathrm{phys}} = 1/16$. The
apparent anisotropy of $\gamma_{ab}$ in primitive reduced coordinates
is exactly the non-orthogonality of the BCC primitive reciprocal basis;
$\gamma_{ab} = (\pi^{2}/4)\,(\boldsymbol{b}_a \cdot \boldsymbol{b}_b)$,
i.e. $\gamma_{ab} = \gamma_{\mathrm{phys}}\,(2\pi)^{2}\,G_{ab}$ where
$G_{ab} = \boldsymbol{b}_a \cdot \boldsymbol{b}_b$ is the reciprocal
metric.

This is sub-target $n_s$-1 of an internal working note. It is
**Need-agnostic**: it does not close $n_s$ itself (the Walker–Curvature
identification remains the structural blocker per
an internal working note §"Need C"), but it establishes a
closed-form mathematical fact about the framework's Bloch operator
near $\Gamma$, sibling to `predictions/srs_E_at_P.py` (P-point) and
`predictions/B_P_doubly_degenerate_h.py` ($B(P)$-spectrum).

## Framework axioms invoked

Inherited from upstream `predictions/` files and theorem docs; no new
axioms introduced here.

- **(A1)** Binary self-inverse toggle (via `predictions/p_toggle.py`,
  `predictions/k_star.py`).
- **(A2)** MDL compression (via `predictions/d_spatial.py`,
  `predictions/g_girth_derivation.md` §2, `predictions/k_star.py`).

The additional framework content used below is itself theorem-grade:

- `predictions/k_star.py` — $k^{*} = 3$.
- `predictions/d_spatial.py` — $d = 3$.
- `predictions/g_girth_derivation.md` §2 — the MDL-optimal 3-regular
  3D crystal net is srs in its standard I4$_132$ realisation with
  Wyckoff position 8a, $x = 1/8$ (Sunada 2012).
- `../predictions/walker_dynamics_derivation.md` (W1–W3) — observables on srs are
  spectral statistics of the non-backtracking walker; $A(\boldsymbol{k})$
  is the scalar Bloch fibre on the four-vertex primitive cell.
- `predictions/B_P_doubly_degenerate_h.py` — sibling Bloch theorem at
  the P-point; supplies the **identical** primitive-cell bond list
  used here.

## Cited mathematical theorems

- **Kato 1980**, *Perturbation Theory for Linear Operators*, 2nd ed.,
  Springer Grundlehren **132**, §II.5 Theorem 5.4. Non-degenerate
  Rayleigh–Schrödinger expansion of an isolated simple eigenvalue
  under analytic perturbation: at second order,
  $$E^{(2)} \;=\; \langle v_0 | H_2 | v_0 \rangle \;+\; \sum_{n \neq 0} \frac{\bigl|\langle v_n | H_1 | v_0 \rangle\bigr|^{2}}{E_0 - E_n}.$$
  Equivalent reference: **Reed–Simon 1978**, *Methods of Modern
  Mathematical Physics*, Vol. IV, Theorem XII.13.
- **Perron–Frobenius theorem** (standard; e.g. Horn–Johnson 1985,
  *Matrix Analysis*, Theorem 8.4.4). For an irreducible non-negative
  matrix, the spectral radius is a simple eigenvalue with a strictly
  positive eigenvector. Applied to $A(\Gamma) = J - I$ on the
  connected $K_4$ adjacency, it gives the simple eigenvalue $+3$ with
  eigenvector $\mathbf{1}/2$.
- **Sunada 2012**, *Notices AMS* **59**(2), 208–215 — srs
  identification (imported here via
  `predictions/g_girth_derivation.md` §2).

## Derivation

### Step 1 — Upstream: $k^{*} = 3$, $d = 3$, srs = I4$_132$ + Wyckoff 8a

From `predictions/k_star.py` and `predictions/d_spatial.py`, the
MDL-optimal observer operates on a 3-regular graph embedded in
3-dimensional space. From `predictions/g_girth_derivation.md` §2 the
graph is the srs lattice in its standard Wyckoff-8a realisation with
internal parameter $x = 1/8$; the space group is I4$_132$ (no. 214)
and the point group is $432$. The four-vertex primitive cell has six
undirected nearest-neighbour bonds, listed in
`predictions/B_P_doubly_degenerate_h.py` lines 129–134. The same list
is used here verbatim; the cell tuples are lattice translations
expressed in the **primitive** BCC basis.

### Step 2 — Walker dynamics: scalar Bloch adjacency on srs

By `../predictions/walker_dynamics_derivation.md` (W1–W3), observable dynamics on
srs reduce to non-backtracking walks; the Bloch decomposition (Sunada
2012 §§5–6) gives a fibre $A(\boldsymbol{k})$ acting on $\mathbb{C}^{4}$
(the four primitive-cell vertices) for each $\boldsymbol{k}$ in the
Brillouin zone. The matrix entries are
$$A(\boldsymbol{k})_{ij} \;=\; \sum_{\boldsymbol{\tau}\,\in\,\text{cells}(i,j)} e^{2\pi\,i\,\boldsymbol{k}\cdot\boldsymbol{\tau}},$$
where the sum runs over primitive lattice translations
$\boldsymbol{\tau}$ realising a bond from $v_j$ to $v_i$, and the
Bloch coordinate $\boldsymbol{k} = (k_1, k_2, k_3)$ is conjugate to
the primitive basis (so $\boldsymbol{k}\cdot\boldsymbol{\tau}$ is just
the inner product of integer cell tuples with $(k_1, k_2, k_3)$).

### Step 3 — $A(\Gamma)$ is the $K_4$ adjacency; spectrum $\{+3,\,-1\!\times\!3\}$

At $\boldsymbol{k} = \Gamma = (0,0,0)$ every phase factor is $1$, so
the matrix entry $A(\Gamma)_{ij}$ counts directed lattice translations
realising bonds between $v_j$ and $v_i$. Inspecting the bond list,
each ordered pair $(i, j)$ with $i \neq j$ has exactly one such
translation (up to its inverse), so
$$A(\Gamma) \;=\; \begin{pmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{pmatrix} \;=\; J - I,$$
the adjacency matrix of $K_4$. Its eigenvalues are $\{+3,\,-1,\,-1,\,-1\}$;
$+3$ is simple by Perron–Frobenius (Horn–Johnson 1985 Thm 8.4.4) with
eigenvector $\mathbf{v}_0 = (1,1,1,1)/2$. Sympy verification of all
three claims (Hermiticity of $A(\boldsymbol{k})$, the matrix value
$A(\Gamma) = J - I$, and the spectrum) is in
`predictions/srs_bloch_dispersion_gamma.py` lines 232–253.

### Step 4 — Orthonormal $(-1)$-eigenspace basis

The $(-1)$-eigenspace of $A(\Gamma)$ is the orthogonal complement of
$\mathbf{v}_0$. Choose the orthonormal basis
$$\mathbf{v}_1 = \tfrac{1}{\sqrt{2}}(1,-1,0,0)^{T},\quad \mathbf{v}_2 = \tfrac{1}{\sqrt{6}}(1,1,-2,0)^{T},\quad \mathbf{v}_3 = \tfrac{1}{2\sqrt{3}}(1,1,1,-3)^{T}.$$
Sympy checks $A(\Gamma)\,\mathbf{v}_n = -\mathbf{v}_n$ for $n = 1, 2,
3$ and $\langle \mathbf{v}_m | \mathbf{v}_n \rangle = \delta_{mn}$ in
the script (lines 257–267). Any orthonormal basis would do; the
energy correction $E^{(2)}$ depends only on the spectral projector
onto the $(-1)$-eigenspace, $P_{-1} = \mathbf{1} - \mathbf{v}_0
\mathbf{v}_0^{\dagger}$.

### Step 5 — Taylor expansion of $A(\boldsymbol{k})$ at $\Gamma$

Define the linear and quadratic Taylor coefficients
$$H_1(\boldsymbol{k}) \;=\; \sum_{a=1}^{3} \frac{\partial A}{\partial k_a}\bigg|_{\Gamma}\,k_a,\qquad H_2(\boldsymbol{k}) \;=\; \tfrac{1}{2}\,\sum_{a, b=1}^{3} \frac{\partial^{2} A}{\partial k_a\,\partial k_b}\bigg|_{\Gamma}\,k_a\,k_b.$$
Sympy gives the linear matrices explicitly (e.g.
$\partial A/\partial k_1|_{\Gamma}$ has entries $\pm 2\pi i$ on the
six bond positions and zeros elsewhere; the entry signs follow the
cell tuples). The first-order Rayleigh–Schrödinger correction
$\langle \mathbf{v}_0 | H_1(\boldsymbol{k}) | \mathbf{v}_0 \rangle$
vanishes identically (sympy-verified, line 281), as required because
$A(\boldsymbol{k})$ is real-analytic and the Perron eigenvalue at
$\Gamma$ is the spectral radius (so $\Gamma$ is a maximum and
$\nabla\lambda_0|_{\Gamma} = 0$).

### Step 6 — Second-order non-degenerate Rayleigh–Schrödinger

Because $+3$ is a simple eigenvalue (Step 3), Kato 1980 §II.5 Theorem
5.4 (or equivalently Reed–Simon 1978 Thm XII.13) applies: the second-
order correction to $\lambda_0$ is
$$E^{(2)}(\boldsymbol{k}) \;=\; \underbrace{\langle \mathbf{v}_0 | H_2(\boldsymbol{k}) | \mathbf{v}_0 \rangle}_{\text{Rayleigh quotient piece}} \;+\; \underbrace{\sum_{n=1}^{3} \frac{\bigl|\langle \mathbf{v}_n | H_1(\boldsymbol{k}) | \mathbf{v}_0 \rangle\bigr|^{2}}{E_0 - E_n}}_{\text{level-mixing piece}},$$
with $E_0 - E_n = 3 - (-1) = 4$ for every $n \in \{1, 2, 3\}$. Sympy
evaluates both pieces in closed form:
$$\langle \mathbf{v}_0 | H_2 | \mathbf{v}_0 \rangle \;=\; -\pi^{2}\,\bigl[\,4\,(k_1^{2} + k_2^{2} + k_3^{2}) + 6\,(k_1 k_2 + k_1 k_3 + k_2 k_3)\,\bigr],$$
$$\frac{1}{4}\,\sum_{n=1}^{3} \bigl|\langle \mathbf{v}_n | H_1 | \mathbf{v}_0 \rangle\bigr|^{2} \;=\; \tfrac{\pi^{2}}{2}\,\bigl[\,7\,(k_1^{2} + k_2^{2} + k_3^{2}) + 11\,(k_1 k_2 + k_1 k_3 + k_2 k_3)\,\bigr].$$
Adding (script lines 286–306):
$$\lambda_0^{(2)}(\boldsymbol{k}) \;=\; -\,\frac{\pi^{2}}{2}\,\bigl[\,(k_1^{2} + k_2^{2} + k_3^{2}) + (k_1 k_2 + k_1 k_3 + k_2 k_3)\,\bigr].$$
Reading off the symmetric tensor $\gamma_{ab}$ defined by
$\lambda_0^{(2)} = -\sum_{a,b} \gamma_{ab} k_a k_b$:
$$\boxed{\;\gamma_{ab} \;=\; \frac{\pi^{2}}{2}\,\delta_{ab} \;+\; \frac{\pi^{2}}{4}\,(1 - \delta_{ab}).\;}$$

### Step 7 — Conversion to physical Cartesian wavevector

The primitive BCC lattice with conventional cubic lattice constant
$a = 1$ has primitive direct vectors $\boldsymbol{a}_1 = \tfrac{1}{2}(-1, 1, 1)$,
$\boldsymbol{a}_2 = \tfrac{1}{2}(1, -1, 1)$, $\boldsymbol{a}_3 = \tfrac{1}{2}(1, 1, -1)$;
the dual primitive reciprocal vectors are
$$\boldsymbol{b}_1 = (0, 1, 1),\quad \boldsymbol{b}_2 = (1, 0, 1),\quad \boldsymbol{b}_3 = (1, 1, 0)$$
satisfying $\boldsymbol{a}_i \cdot \boldsymbol{b}_j = \delta_{ij}$
(verifiable by direct computation; sanity-checked via the v_0–v_1 NN
bond length in `predictions/srs_bloch_dispersion_gamma.py` script
comments). The physical (Cartesian) wavevector associated with the
Bloch coordinate $(k_1, k_2, k_3)$ is
$$\boldsymbol{q} \;=\; 2\pi\,(k_1\,\boldsymbol{b}_1 + k_2\,\boldsymbol{b}_2 + k_3\,\boldsymbol{b}_3),$$
and its squared magnitude is $|\boldsymbol{q}|^{2} = (2\pi)^{2}\,k^{T} G\,k$
with reciprocal metric
$$G_{ab} \;=\; \boldsymbol{b}_a \cdot \boldsymbol{b}_b \;=\; \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix}_{ab}.$$
Comparing with Step 6,
$$\gamma_{ab} \;=\; \tfrac{\pi^{2}}{4}\,G_{ab} \;=\; \tfrac{1}{16}\,(2\pi)^{2}\,G_{ab},$$
i.e.
$$\sum_{a,b}\gamma_{ab} k_a k_b \;=\; \tfrac{1}{16}\,|\boldsymbol{q}|^{2}.$$
Therefore
$$\lambda_0(\boldsymbol{q}) \;=\; k^{*} \;-\; \tfrac{1}{16}\,|\boldsymbol{q}|^{2} \;+\; O(|\boldsymbol{q}|^{4}),$$
i.e. the dispersion is isotropic in physical Cartesian wavevector
with the single closed scalar
$$\gamma_{\mathrm{phys}} \;=\; \frac{1}{16}.$$
The "anisotropy" of $\gamma_{ab}$ is purely the non-orthogonality
$\boldsymbol{b}_a \cdot \boldsymbol{b}_b = 1 \neq 0$ of the primitive
BCC reciprocal basis. Cubic isotropy is restored by the factor of $G$.

### Step 8 — Numerical cross-checks

(a) Cartesian isotropy. Picking $|\boldsymbol{q}| = 10^{-3}$ in seven
Cartesian directions (principal axes, face-diagonal, body-diagonal,
and an off-symmetry $(3,-1,2)/\sqrt{14}$), converting to primitive
reduced $(k_1, k_2, k_3) = (2\pi)^{-1} M^{-1} \boldsymbol{q}$ with
$M$ the column matrix of $\boldsymbol{b}_i$, and diagonalising
$A(\boldsymbol{k})$ in double precision: the ratio
$(3 - \lambda_0(\boldsymbol{k})) \,\big/\, (|\boldsymbol{q}|^{2}/16)$
equals $1.000000$ to six printed digits in every direction (script
lines 326–349; reported in the script's "Numerical check" block).

(b) $O(|\boldsymbol{k}|^{4})$ scaling. With $\boldsymbol{k} = (\varepsilon, 0, 0)$
in primitive reduced coordinates, the residual
$3 - \lambda_0(\boldsymbol{k}) - (\pi^{2}/2)\,\varepsilon^{2}$ scales
as $\varepsilon^{4}$ for $\varepsilon \in \{10^{-2}, 5\!\cdot\!10^{-3},
10^{-3}, 5\!\cdot\!10^{-4}\}$: the ratio
$\text{residual}/\varepsilon^{4}$ stabilises near $-4.06$ (script lines
352–360), confirming the second-order RS truncation is correct and the
next non-trivial correction is at quartic order.

## Result

$$\boxed{\;\lambda_0(\boldsymbol{k}) \;=\; k^{*} \;-\; \sum_{a,b} \gamma_{ab}\,k_a\,k_b \;+\; O(|\boldsymbol{k}|^{4}), \qquad \gamma_{ab} \;=\; \tfrac{\pi^{2}}{2}\delta_{ab} + \tfrac{\pi^{2}}{4}(1-\delta_{ab});\;}$$

equivalently, in physical Cartesian wavevector,
$$\lambda_0(\boldsymbol{q}) \;=\; k^{*} \;-\; \tfrac{1}{16}\,|\boldsymbol{q}|^{2} \;+\; O(|\boldsymbol{q}|^{4}), \qquad \gamma_{\mathrm{phys}} \;=\; \tfrac{1}{16}.$$

Numerical values for $k^{*} = 3$:

| Quantity | Value (sympy / exact) | Value (decimal) |
|---|---|---|
| $\gamma_{aa}$ | $\pi^{2}/2$ | $4.93480\ldots$ |
| $\gamma_{ab}$ ($a \neq b$) | $\pi^{2}/4$ | $2.46740\ldots$ |
| $\gamma_{\mathrm{phys}}$ | $1/16$ | $0.0625$ |

The pure function `predict_srs_bloch_dispersion_gamma(k_star)` returns
the pair $(\gamma_{ab},\,\gamma_{\mathrm{phys}})$ for $k^{*} = 3$ and
raises for any other input.

## Comparison with "observation"

There is no physical parameter to fit; the statement is a structural
identity of the Bloch scalar adjacency on srs. "Observation" here is
the numerical Perron eigenvalue of $A(\boldsymbol{k})$ for small
$\boldsymbol{k}$:

| Quantity | Symbolic | Numerical (at $|\boldsymbol{q}| = 10^{-3}$) | Deviation |
|---|---|---|---|
| $3 - \lambda_0(\boldsymbol{q})\,/\,(|\boldsymbol{q}|^{2}/16)$ | $1$ | $1.000000$ (every direction) | $< 10^{-6}$ at this $|\boldsymbol{q}|$; converges as $|\boldsymbol{q}|^{2}$ |
| $\bigl|\text{residual at }\varepsilon=10^{-3}\bigr|/\varepsilon^{4}$ | $\sim O(1)$ | $4.06$ | confirms quartic next-order |

## Per-step gate-clear types

Against the parameter-linter hard gate (1 axiom / 2 explicit algebra /
3 cited theorem / 4 upstream closed file):

| Step | Content | Gate type |
|---|---|---|
| 1 | $k^{*} = 3$, $d = 3$, srs embedding | 4 (upstream) |
| 2 | Scalar Bloch adjacency $A(\boldsymbol{k})$ on srs primitive cell | 4 (`../predictions/walker_dynamics_derivation.md`; `B_P_doubly_degenerate_h.py`) |
| 3 | $A(\Gamma) = J - I$; spectrum $\{+3,\,-1\!\times\!3\}$; $+3$ Perron-simple | 2 (sympy) + 3 (Perron–Frobenius, Horn–Johnson 1985 Thm 8.4.4) |
| 4 | Orthonormal $(-1)$-eigenspace basis | 2 (sympy) |
| 5 | Taylor expansion $H_1, H_2$; first-order vanishes | 2 (sympy) |
| 6 | Second-order RS energy; closed-form $\gamma_{ab}$ | 2 (sympy) + 3 (Kato 1980 §II.5 Thm 5.4) |
| 7 | Cartesian-wavevector conversion; isotropic $\gamma_{\mathrm{phys}} = 1/16$ | 2 (sympy + reciprocal-metric algebra) |
| 8 | Numerical isotropy + $O(|\boldsymbol{k}|^{4})$ scaling | 2 (numerical) |

Every step is one of types 1–4. No step is "it follows structurally";
no step selects an alternative by fit; no step imports a
phenomenological input.

## Open questions

1. **Closed-form sum-over-bonds expression for $\gamma_{ab}$.** The
   Rayleigh-quotient piece $\langle \mathbf{v}_0 | H_2 | \mathbf{v}_0 \rangle$
   admits the simple sum-over-directed-edges formula
   $$\langle \mathbf{v}_0 | H_2 | \mathbf{v}_0 \rangle \;=\; -\,\frac{(2\pi)^{2}}{2 N}\sum_{e\,\text{directed}} (\boldsymbol{c}_e \cdot \boldsymbol{k})^{2} \cdot v_0[e_{\text{src}}]\,v_0[e_{\text{tgt}}]$$
   with $N = 4$ (primitive-cell vertex count) and $\boldsymbol{c}_e$
   the cell tuple of edge $e$. Sympy verifies it gives the
   "diagonal-term" expression in Step 6. The level-mixing piece does
   not factorise into a single sum over edges — it requires the spectral
   projector $P_{-1}$ — but for the Perron-uniform $\mathbf{v}_0$, both
   pieces conspire to halve the bare Rayleigh anisotropy and produce
   the exact reciprocal-metric structure $\gamma_{ab} \propto G_{ab}$.
   A purely combinatorial derivation of the factor of $1/16$ from the
   Wyckoff-8a bond geometry alone (without invoking the perturbative
   level mixing) would be a strengthening; not a gap.

2. **Relation to $n_s$.** This lemma only computes the small-
   $\boldsymbol{k}$ shape of $\lambda_0(\boldsymbol{k})$. To convert
   into a primordial scalar power-spectrum slope $n_s - 1$ requires the
   load-bearing structural identifications C (Walker–Curvature) and D
   (quantization rule) of an internal working note, both of which
   remain open. The shape exponent of $|\boldsymbol{q}|^{2}$ here would,
   if naively read as $n_s - 1$, give $n_s = 3$, not the observed
   $n_s \approx 0.965$. The lemma is therefore Need-agnostic and
   does **not** close $n_s$.

3. **Bloch decomposition.** The decomposition $A = \int^{\oplus}
   A(\boldsymbol{k})\,d\boldsymbol{k}$ rests on standard
   crystallographic Fourier theory (Sunada 2012 §§5–6); it is treated
   here as an established citation, not re-proved. Same status as
   `predictions/B_P_doubly_degenerate_h.py` Step 2.


## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.

## References

- Kato, T. (1980). *Perturbation Theory for Linear Operators*, 2nd ed.
  Springer Grundlehren der mathematischen Wissenschaften **132**.
  §II.5 Theorem 5.4 (analytic perturbation of an isolated simple
  eigenvalue).
- Reed, M. & Simon, B. (1978). *Methods of Modern Mathematical Physics*,
  Vol. IV: *Analysis of Operators*. Academic Press. Theorem XII.13.
- Horn, R. A. & Johnson, C. R. (1985). *Matrix Analysis*. Cambridge
  University Press. Theorem 8.4.4 (Perron–Frobenius).
- Sunada, T. (2012). Lecture on topological crystallography.
  *Notices AMS* **59**(2), 208–215.
- Ashcroft, N. W. & Mermin, N. D. (1976). *Solid State Physics*. Holt,
  Rinehart and Winston. Chapter 10 (tight-binding) for context — the
  expansion form $\lambda_0 = k^{*} - \gamma\,|\boldsymbol{q}|^{2}$ is
  standard in band theory; the computation here is the explicit srs
  realisation of the closed-form coefficient.
- O'Keeffe, M., Peskov, M. A., Ramsden, S. J., & Yaghi, O. M. (2008).
  The Reticular Chemistry Structure Resource (RCSR) database.
  *Accts. Chem. Res.* **41**, 1782–1789. Entry `srs`.

## Files referenced

- `predictions/k_star.py`, `predictions/d_spatial.py`,
  `predictions/g_girth_derivation.md` §2 — upstream.
- `../predictions/walker_dynamics_derivation.md` — upstream.
- `predictions/B_P_doubly_degenerate_h.py`,
  `predictions/B_P_doubly_degenerate_h_derivation.md` — sibling
  Bloch theorem at $P$; supplies the identical primitive-cell bond list
  used here.
- `predictions/srs_E_at_P.py` — closest sibling (P-point Perron eigenvalue).
- `predictions/srs_cubic_moment_derivation.md` — sibling theorem-grade
  derivation template; uses the same Wyckoff-8a + 432 setup.
  $n_s$-1).

## Verification

```
python3 predictions/srs_bloch_dispersion_gamma.py
```

Expected final line: `OK: outputs agree.  lambda_0(q) = k* - |q|^2/16 + O(|q|^4) on srs.`
