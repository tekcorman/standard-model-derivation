# Derivation: scalar Bloch H Perron-band dim-6 Lorentz violation coefficients

**Audit anchor:** Foundational Lorentz-arc result. Conditional on Rows 4, 6 of `docs/audits/registers/uniqueness_ledger.md` (k* = 3 + srs identification). Theorem-grade SYMBOLIC via Feshbach-Löwdin. Component of the joint LORENTZ_SIG closure per `docs/theorems/lorentz_sig_ccclose_joint_closure.md`.

## Abstract

The 4-band scalar adjacency Bloch operator $H(\mathbf k)$ on srs (I4₁32 +
Wyckoff 8a + nearest-neighbour bonds) has Perron eigenvalue $\lambda_0(\mathbf k)$
with small-$|\mathbf k|$ Taylor expansion

$$\lambda_0(\mathbf k) \;=\; 3 \;-\; D_H\,|\mathbf k|^2 \;-\; \bigl[D_4^{\rm iso} + D_4^{\rm aniso}\, f_4(\hat{\mathbf k})\bigr]\,|\mathbf k|^4 \;+\; O(|\mathbf k|^6),$$

with $f_4(\hat{\mathbf k}) = \hat k_x^4 + \hat k_y^4 + \hat k_z^4$ the cubic-anisotropy function.
We establish the closed-form values

$$\boxed{\quad D_H = \tfrac{1}{16},\qquad D_4^{\rm iso} = -\tfrac{1}{1024},\qquad D_4^{\rm aniso} = +\tfrac{1}{1536},\qquad \eta^H_{\rm NB} := \frac{D_4^{\rm aniso}}{D_H^2} = \tfrac{1}{6}. \quad}$$

All four values are **theorem-grade symbolic** under the parameter_linter hard
quality gate. The quadratic coefficient $D_H = 1/16$ was already established
via second-order Kato (S3) in `predictions/srs_bloch_dispersion_gamma.py`. The
two quartic coefficients are derived by symbolic Feshbach-Löwdin partition of
$H(\mathbf k)$ relative to the non-degenerate Perron eigenstate, with the
fixed-point equation iterated to convergence in `sympy` exact arithmetic
(`proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py`). The earlier
high-precision numerical extraction
(`proofs/foundations/lorentz_sig_h_lv_coefficients.py`, 25+ digit Vandermonde
at 500-bit `mpmath`) is now an independent cross-check of the symbolic result.

The headline LV ratio $\eta^H_{\rm NB} = 1/6$ is **exactly twice** the existing
framework prediction for the Hashimoto walker ($\eta_{\rm NB} = 1/12$;
`predictions/eta_lattice_lorentz_dim6.py`). The factor of 2 has a clean
structural origin in the Ihara factorisation $u^2 - \lambda u + 2 = 0$ for
3-regular graphs, with derivative $u'(3) = 2$ at the Perron eigenvalue.

## Framework axioms invoked

- **A1** (binary self-inverse toggle): srs adjacency $A_{\rm srs}$ via NN walker.
- **A2-T** (waterline theorem): selects the I4₁32 + Wyckoff 8a substrate uniquely
  (`predictions/g_girth_derivation.md` §2).
- **A3** (complex Hilbert): $H(\mathbf k) \in \mathbb C^{4\times 4}$ Hermitian.

The upstream substrate identification (`predictions/k_star.py`,
`predictions/d_spatial.py`, `predictions/g_girth_derivation.md`,
Sunada 2012 *Topological Crystallography*) is theorem-grade.

## Cited theorems

- **Kato 1980** *Perturbation Theory for Linear Operators* §II.4 (Schur–Feshbach /
  Löwdin partition for non-degenerate eigenvalues) and §II.5 Thm 5.4
  (Rayleigh-Schrödinger expansion). Both used; the quartic coefficients are
  obtained via §II.4 partition iterated symbolically.
- **Ihara 1966** / **Stark–Terras 1996** (Ihara zeta function for $k$-regular
  graphs): scalar adjacency eigenvalues $\lambda$ and Hashimoto eigenvalues $u$
  related by $u^2 - \lambda u + (k-1) = 0$; for $k = 3$ this is
  $u^2 - \lambda u + 2 = 0$.
- **Biggs 1993** *Algebraic Graph Theory* §2.2: $\mathrm{spec}(A_{K_4}) = \{3, -1, -1, -1\}$;
  Perron eigenvalue $+3$ is non-degenerate with eigenvector $\mathbf v_0 = (1,1,1,1)/2$.

## Derivation

### Step 1 — Quadratic coefficient $D_H = 1/16$ (S3)

By Kato §II.5 Thm 5.4 applied to the Perron eigenvalue of the K_4 adjacency at
$\Gamma$ (eigenvalue $+3$, eigenvector $\mathbf v_0 = (1,1,1,1)/2$), the
second-order correction to $\lambda_0$ is

$$\lambda_0^{(2)}(\mathbf k) \;=\; \langle \mathbf v_0|\,V_2(\mathbf k)\,|\mathbf v_0\rangle \;+\; \sum_{m\neq 0} \frac{|\langle m|\,V_1(\mathbf k)\,|\mathbf v_0\rangle|^2}{3 - \lambda_m}.$$

For srs, $V_1$ is the linear-in-$\mathbf k$ Taylor coefficient of $H(\mathbf k)$
and the summed-over states $|m\rangle$ are the three $\lambda = -1$ basis states
of $\mathbf v_0^\perp$, with energy denominator $3 - (-1) = 4$. Direct computation
(Sec II of `predictions/srs_bloch_dispersion_gamma_derivation.md`) gives

$$D_H \;=\; \tfrac{1}{16}\quad\text{(Cartesian-isotropic)}.$$

### Step 2 — Symbolic Feshbach-Löwdin derivation of the full Taylor expansion

The 4-band Bloch Hamiltonian decomposes relative to the non-degenerate Perron
eigenstate $\mathbf v_0 = (1,1,1,1)/2$ via the Schur-Feshbach partition (Kato §II.4).
Writing $P = \mathbf v_0 \mathbf v_0^\dagger$ and $Q = I - P$:

$$
H(\mathbf k) \;=\; \begin{pmatrix} H_{PP}(\mathbf k) & H_{PQ}(\mathbf k) \\ H_{QP}(\mathbf k) & H_{QQ}(\mathbf k) \end{pmatrix}
$$

with $H_{PP}(0) = 3$ and $H_{QQ}(0) = -I_3$ from the K_4 spectrum (Biggs 1993).
The Perron eigenvalue $\lambda_0(\mathbf k)$ satisfies the **exact** equation

$$
\lambda_0(\mathbf k) \;=\; H_{PP}(\mathbf k) \;+\; H_{PQ}(\mathbf k)\,\bigl[\lambda_0\,I_3 - H_{QQ}(\mathbf k)\bigr]^{-1}\,H_{QP}(\mathbf k).
$$

Setting $\delta(\mathbf k) = \lambda_0(\mathbf k) - 3$ and $w_{QQ}(\mathbf k) = H_{QQ}(\mathbf k) + I_3$ (so that $w_{QQ}(0) = 0$), the equation becomes the fixed-point form

$$
\delta(\mathbf k) \;=\; \bigl[H_{PP}(\mathbf k) - 3\bigr] \;+\; H_{PQ}(\mathbf k)\,\bigl[(4 + \delta)\,I_3 - w_{QQ}(\mathbf k)\bigr]^{-1}\,H_{QP}(\mathbf k).
$$

Truncating $H(\mathbf k)$ to total degree 4 in $(k_x, k_y, k_z)$ via the exact
Taylor series of $\exp(i\mathbf k \cdot \mathbf r_{\rm disp}) = \sum_{m=0}^{4} (i\mathbf k\cdot\mathbf r)^m / m!$
on rational displacements $\mathbf r_{\rm disp} = \mathbf r_\beta + \mathbf n\cdot \mathbf A_{\rm prim} - \mathbf r_\alpha$
(with $\mathbf r_\alpha$ the Wyckoff 8a positions $(\tfrac18,\tfrac18,\tfrac18)$, etc.,
and $\mathbf A_{\rm prim}$ the BCC primitive vectors with components $\pm\tfrac12$)
gives a polynomial $H(\mathbf k)$ with exact-rational coefficients. The
Neumann-series expansion

$$
\bigl[(4+\delta) I_3 - w_{QQ}\bigr]^{-1} \;=\; \frac{1}{4+\delta}\sum_{n\ge 0}\Bigl(\frac{w_{QQ}}{4+\delta}\Bigr)^{n}
$$

truncated to total degree 4 in $\mathbf k$ closes the iteration in `sympy`
exact arithmetic. Convergence is reached in **2 iterations** because (i) the
linear-in-$\mathbf k$ contribution to $\delta$ vanishes by inversion symmetry
($\langle \mathbf v_0 | V_1(\mathbf k) | \mathbf v_0\rangle = 0$ since each
undirected bond contributes $\pm i \mathbf k\cdot \mathbf r$ in pairs), so
$\delta = O(\mathbf k^2)$ at minimum; and (ii) the resolvent's $\delta$-feedback
is then $O(\mathbf k^2)$ and only enters at order $\mathbf k^4$ via the sandwich
$H_{PQ}\,(\delta/16)\,H_{QP}$.

The full 4th-order Taylor result, in physical Cartesian $\mathbf k$
(matching the gauge of `proofs/foundations/lorentz_sig_h_lv_coefficients.py`), is

$$
\delta(\mathbf k) \;=\; -\tfrac{1}{16}(k_x^2 + k_y^2 + k_z^2) \;+\; \tfrac{1}{3072}(k_x^4 + k_y^4 + k_z^4) \;+\; \tfrac{1}{512}(k_x^2 k_y^2 + k_x^2 k_z^2 + k_y^2 k_z^2) \;+\; O(\mathbf k^6).
$$

Decomposing the quartic part in the cubic-symmetric basis
$\alpha\,(k_x^2+k_y^2+k_z^2)^2 + \beta\,(k_x^4+k_y^4+k_z^4)$:

- coefficient of $k_x^4$: $\alpha + \beta = \tfrac{1}{3072}$
- coefficient of $k_x^2 k_y^2$: $2\alpha = \tfrac{1}{512}$, so $\alpha = \tfrac{1}{1024}$ and $\beta = -\tfrac{1}{1536}$.

Matching against the dispersion convention $\delta = -D_H\,|\mathbf k|^2 - D_4^{\rm iso}\,|\mathbf k|^4 - D_4^{\rm aniso}\,(k_x^4 + k_y^4 + k_z^4)$:

$$\boxed{\;D_H \;=\; \tfrac{1}{16},\qquad D_4^{\rm iso} \;=\; -\alpha \;=\; -\tfrac{1}{1024},\qquad D_4^{\rm aniso} \;=\; -\beta \;=\; +\tfrac{1}{1536}.\;}$$

This is **fully symbolic**: every step uses exact-rational arithmetic, no
numerical fit, no dimension-of-rational-space coincidence argument. Verified
end-to-end by `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py`
(runs in ~4 seconds, asserts the closed-form rationals).

The earlier 25+ digit numerical extraction
(`proofs/foundations/lorentz_sig_h_lv_coefficients.py`, three high-symmetry
directions $[100], [110], [111]$ with $f_4 = \{1, 1/2, 1/3\}$, 4-point Vandermonde
fit at 500-bit `mpmath` precision) now serves as an **independent cross-check**
of the symbolic result rather than the primary source of the rational values.

### Step 3 — Headline LV ratio $\eta^H_{\rm NB} = 1/6$

By definition

$$\eta^H_{\rm NB} \;=\; \frac{D_4^{\rm aniso}}{D_H^2} \;=\; \frac{1/1536}{(1/16)^2} \;=\; \frac{1}{1536} \cdot 256 \;=\; \frac{256}{1536} \;=\; \frac{1}{6}.$$

Sign: $+1/6 > 0$, subluminal (cubic anisotropy slows propagation along $[100]$
relative to $[111]$, since $f_4([100]) = 1 > f_4([111]) = 1/3$ contributes to
the negative quartic correction).

### Step 4 — Ihara cross-walker theorem (consistency check)

For 3-regular graphs the Ihara factorization gives Hashimoto eigenvalue $h(\lambda)$
in terms of scalar adjacency $\lambda$ via $u^2 - \lambda u + 2 = 0$, with the upper
root

$$h(\lambda) \;=\; \tfrac{1}{2}\bigl(\lambda + \sqrt{\lambda^2 - 8}\bigr).$$

Direct differentiation gives $h'(3) = 2$, $h''(3) = -4$. Substituting the scalar
Taylor expansion $\lambda_0(\mathbf k) - 3 = -D_H k^2 - \alpha^H k^4 + O(k^6)$ into
$h(\lambda_0(\mathbf k)) - 2$ and matching against
$h_{\max}(\mathbf k) - 2 = -D_{\rm NB} k^2 - \alpha^{\rm NB} k^4 + O(k^6)$ gives
the **Ihara cross-walker relations**:

$$D_{\rm NB} \;=\; h'(3)\,D_H \;=\; 2\cdot \tfrac{1}{16} \;=\; \tfrac{1}{8},\qquad
D_4^{{\rm aniso},{\rm NB}} \;=\; h'(3)\,D_4^{\rm aniso} \;=\; 2\cdot \tfrac{1}{1536} \;=\; \tfrac{1}{768},$$

$$D_4^{{\rm iso},{\rm NB}} \;=\; h'(3)\,D_4^{\rm iso} \;-\; \tfrac{1}{2}\,h''(3)\,D_H^2 \;=\; 2\cdot\bigl(-\tfrac{1}{1024}\bigr) \;-\; \tfrac{1}{2}\cdot(-4)\cdot \tfrac{1}{256} \;=\; -\tfrac{1}{512} + \tfrac{1}{128} \;=\; +\tfrac{3}{512},$$

$$\eta_{\rm NB} \;=\; \frac{D_4^{{\rm aniso},{\rm NB}}}{D_{\rm NB}^2} \;=\; \frac{2\,D_4^{\rm aniso}}{(2\,D_H)^2} \;=\; \frac{1}{2}\,\eta^H_{\rm NB} \;=\; \tfrac{1}{12}.$$

The first three Hashimoto values $D_{\rm NB} = 1/8$, $D_4^{{\rm aniso},{\rm NB}} = 1/768$,
$\eta_{\rm NB} = 1/12$ all match the existing framework predictions in
`predictions/eta_lattice_lorentz_dim6.py` and
`proofs/lorentz/hashimoto_dispersion_symbolic.py`. The fourth value
$D_4^{{\rm iso},{\rm NB}} = +3/512$ is a new prediction; it is independently
verified by direct numerical extraction on the Hashimoto operator
(`proofs/foundations/lorentz_sig_hashimoto_d4_iso.py` matches at 25+ digits).

This is a **non-trivial consistency check** for the closed-form values
$D_4^{\rm iso} = -1/1024$ and $D_4^{\rm aniso} = +1/1536$: they not only fit the
scalar-Bloch numerical extraction but also reproduce three pre-existing
Hashimoto values via Ihara, and predict a fourth that's verified.

## Result

$$\boxed{\quad D_H = \tfrac{1}{16},\qquad D_4^{\rm iso} = -\tfrac{1}{1024},\qquad D_4^{\rm aniso} = +\tfrac{1}{1536},\qquad \eta^H_{\rm NB} = \tfrac{1}{6}. \quad}$$

All numerical: $D_H = 0.0625$ exactly; $D_4^{\rm iso} \approx -9.766\times 10^{-4}$,
$D_4^{\rm aniso} \approx +6.510\times 10^{-4}$, $\eta^H_{\rm NB} = 0.16\overline{6}$.

## Comparison with experiment

N/A — these are structural quantities for the scalar Bloch operator. The framework's
"observable" LV channel runs through the Hashimoto walker $\eta_{\rm NB} = 1/12$
(`predictions/eta_lattice_lorentz_dim6.py`), $\sim 16$ orders of magnitude below
current LIV bounds (LHAASO GRB 221009A et al.). The scalar-Bloch sister value
$\eta^H_{\rm NB} = 1/6$ is therefore not directly measurable; its physics relevance
is structural (Ihara cross-walker relation underpinning the Hashimoto value).

## Open questions

1. **Multi-valley physics at sub-leading order.** The dim-6 LV coefficients here
   are LOCAL Kato properties at $\Gamma$; sub-dominant cones at $H, P$ do NOT
   contribute (an earlier conjecture in
   an internal working note was refuted on this
   point). Sub-dominant cones are physically relevant only at the global
   continuum-limit lift (research-level, item 6 of multi-valley scoping).

## References

- Biggs, N. (1993). *Algebraic Graph Theory*, 2nd ed. Cambridge Univ. Press.
- Kato, T. (1980). *Perturbation Theory for Linear Operators*, 2nd ed.,
  Springer Grundlehren **132**.
- Ihara, Y. (1966). On discrete subgroups of the two-by-two projective linear
  group over p-adic fields. *J. Math. Soc. Japan* **18**, 219.
- Stark, H. M., Terras, A. A. (1996). Zeta functions of finite graphs and
  coverings. *Adv. Math.* **121**, 124.
- Sunada, T. (2012). *Topological Crystallography: With a View Towards Discrete
  Geometric Analysis*, Springer.
- `predictions/k_star.py`, `predictions/d_spatial.py`,
  `predictions/g_girth_derivation.md` — substrate identification at theorem grade.
- `predictions/srs_bloch_dispersion_gamma.py` — quadratic dispersion theorem at $\Gamma$
  ($D_H = 1/16$ via 2nd-order Kato).
- `predictions/eta_lattice_lorentz_dim6.py` — Hashimoto sister, $\eta_{\rm NB} = 1/12$.
- `predictions/srs_dirac_cone_velocities.py` — spin-1 Dirac at the Γ cone.
- `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py` — symbolic
  Feshbach-Löwdin / Rayleigh-Schrödinger derivation (the primary source of the
  closed-form rationals; runs in ~4 seconds with `sympy` exact arithmetic).
- `proofs/foundations/lorentz_sig_h_lv_coefficients.py` — high-precision numerical
  cross-check (4-point Vandermonde at 500-bit `mpmath`, three directions).
- `proofs/foundations/lorentz_sig_ihara_lv_relation.py` — symbolic Ihara
  cross-walker theorem.
- `proofs/foundations/lorentz_sig_hashimoto_d4_iso.py` — independent numerical
  verification of $D_4^{{\rm iso},{\rm NB}} = +3/512$.
- `proofs/lorentz/hashimoto_dispersion_symbolic.py` — sister extraction for the
  Hashimoto walker (D_NB, D4_aniso).
  + Ihara cross-walker theorem section + global-lift research path.
