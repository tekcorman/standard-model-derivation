# Derivation of the srs cubic moment formula

## Abstract

We derive the identity

$$\left\langle (\mathbf{e}\cdot\hat{\mathbf{z}})^{2n} \right\rangle \;=\; \frac{1}{k^{*}\,2^{\,n-1}} \qquad (n \geq 1)$$

for the 2n-th power of the inner product between a unit directed bond vector $\mathbf{e}$ of the srs lattice and a principal cubic axis $\hat{\mathbf{z}}$, averaged over the $N_e = 24$ directed nearest-neighbor edges of the conventional cubic cell of the $I4_132$ realization of srs. The averaging identity is equivalent to a partition of the 24 edges into 8 perpendicular edges $((\mathbf{e}\cdot\hat{\mathbf{z}})^2 = 0)$ and 16 diagonal edges $((\mathbf{e}\cdot\hat{\mathbf{z}})^2 = \tfrac{1}{2})$, a consequence of the fact that the srs bond vectors are exactly the 12 $\langle 110\rangle$ face-diagonal directions (each appearing in two orientations). Every step is either an explicit arithmetic/geometric step or an upstream result in the `predictions/` directory; the only classical citation used is the srs identification of Sunada (2012).

## Framework axioms invoked

Inherited from upstream predictions files — no new axioms introduced here:

- **MDL compression** (via `predictions/d_spatial.py` and `predictions/g_girth_derivation.md`).
- **Self-inverse binary toggle** (via `predictions/p_toggle.py` and `predictions/k_star.py`).

## Derivation

### Step 1. Upstream: $k^{*} = 3$, $d = 3$

From `predictions/k_star.py` and `predictions/d_spatial.py`, the MDL-optimal observer operates on a 3-regular graph embedded in a 3-dimensional space. These are closed upstream results under A1 + A2-T (the structural axiom A1 plus the MDL waterline theorem; see `docs/framework/framework_axioms.md` §10 and `docs/theorems/theorem_A2_mdl_from_finite_register.md`).

### Step 2. Upstream: srs lattice identification

From `predictions/g_girth_derivation.md` §2 ("srs is the unique MDL minimum among 3-regular 3D crystal nets" — sharp-peak case, waterline = strict-min agree by Sunada uniqueness; not subject to the `canonical_encoding`/`channel_select` operator split), the observer's optimal graph is the srs (Laves) net. The key input is Sunada's uniqueness theorem:

> **Theorem (Sunada, *Notices AMS* **59**(2), 208–215, 2012).** The srs (Laves) lattice is the unique 3-connected 3D crystal net that is both vertex-transitive and edge-transitive.

Combined with the three-case description-length analysis in `proofs/foundations/dl_comparison.py` (Cases 1–3 in `g_girth_derivation.md`), this fixes the graph up to isomorphism and fixes its embedding as the standard realization cataloged in RCSR (O'Keeffe, Peskov, Ramsden & Yaghi, *Accts. Chem. Res.* **41**, 1782–1789, 2008, entry `srs`):

- Space group $I4_132$ (No. 214).
- Point group $O$ = 432 (the chiral cubic rotation group, order 24).
- Vertex positions at Wyckoff 8a with internal parameter $x = 1/8$.

The conventional cubic unit cell contains 8 vertices at
$$
v_{0\dots 3} \;=\; \left(\tfrac18,\tfrac18,\tfrac18\right),\;\left(\tfrac38,\tfrac78,\tfrac58\right),\;\left(\tfrac78,\tfrac58,\tfrac38\right),\;\left(\tfrac58,\tfrac38,\tfrac78\right),
$$
together with the body-centered translates $v_{4\dots7} = v_{0\dots 3} + (\tfrac12,\tfrac12,\tfrac12)\ \bmod\ 1$. These eight positions are the standard Wyckoff-8a realization used by RCSR and Sunada.

### Step 3. Nearest-neighbor bond vectors lie along $\langle 110\rangle$

We compute the nearest-neighbor displacements from the Wyckoff coordinates above, working in the conventional cubic lattice with parameter $a = 1$ and taking minimum-image displacements modulo the lattice translations $\mathbb{Z}^3$.

Consider $v_0 = (1/8,1/8,1/8)$. Its three nearest neighbors are $v_4 = v_0 + (1/2,1/2,1/2)$, and two of the body-centered vertices in the $(\pm\tfrac14,\pm\tfrac14,\pm\tfrac14)$ neighborhood of $v_0$. After the minimum-image reduction, the three displacement vectors at $v_0$ are
$$
\mathbf{d}_1 \;=\; \tfrac14(-1,+1,0), \qquad \mathbf{d}_2 \;=\; \tfrac14(+1,0,-1), \qquad \mathbf{d}_3 \;=\; \tfrac14(0,-1,+1),
$$
each of length $|\mathbf{d}_i| = a\sqrt{2}/4$. Normalising,
$$
\hat{\mathbf{e}}_1 = \tfrac{1}{\sqrt{2}}(-1,+1,0), \quad \hat{\mathbf{e}}_2 = \tfrac{1}{\sqrt{2}}(+1,0,-1), \quad \hat{\mathbf{e}}_3 = \tfrac{1}{\sqrt{2}}(0,-1,+1).
$$

All three lie along the $\langle 110\rangle$ family of face-diagonal directions. Applying the remaining 7 space-group generators to this local frame, and then taking both orientations of each undirected edge, the 24 directed bond vectors of the conventional cell cover each of the 12 $\langle 110\rangle$ unit vectors exactly twice. This is verified numerically by `proofs/flavor/srs_bloch_hamiltonian.py` (`build_unit_cell`, `find_connectivity`), which returns 24 directed bonds with unit vectors
$$
\left\{\tfrac{1}{\sqrt2}(\pm 1,\pm 1, 0), \; \tfrac{1}{\sqrt2}(\pm 1, 0, \pm 1), \; \tfrac{1}{\sqrt2}(0, \pm 1, \pm 1)\right\},
$$
each with multiplicity 2.

### Step 4. Edge count: 24 directed edges per conventional cell

Eight vertices per conventional cell, each with coordination number $k^{*} = 3$, gives
$$
N_e \;=\; 8 \cdot 3 \;=\; 24
$$
directed edges. Equivalently, 12 undirected edges each counted in two orientations. This is an arithmetic statement; $k^{*} = 3$ is supplied upstream.

### Step 5. Projection partition under a principal cubic axis

Let $\hat{\mathbf{z}} = (0,0,1)$ (the argument is identical for $(1,0,0)$ and $(0,1,0)$ by the $S_3$ permutation symmetry of the cubic axes within the 432 point group). The 12 $\langle 110\rangle$ unit vectors partition as:

| Subset | $\langle 110\rangle$ vectors | $(\mathbf{e}\cdot\hat{\mathbf{z}})^2$ |
|--------|------------------------------|----------------------------------------|
| 4 lines in the $xy$-plane | $\tfrac{1}{\sqrt2}(\pm 1, \pm 1, 0)$ | $0$ |
| 4 lines in the $xz$-plane | $\tfrac{1}{\sqrt2}(\pm 1, 0, \pm 1)$ | $1/2$ |
| 4 lines in the $yz$-plane | $\tfrac{1}{\sqrt2}(0, \pm 1, \pm 1)$ | $1/2$ |

Doubling to directed edges: **8 directed edges with $(\mathbf{e}\cdot\hat{\mathbf{z}})^2 = 0$** and **16 directed edges with $(\mathbf{e}\cdot\hat{\mathbf{z}})^2 = 1/2$**. This is an explicit enumeration from Step 3, not an appeal to symmetry.

### Step 6. The moment formula

Using the partition from Step 5,
$$
\left\langle (\mathbf{e}\cdot\hat{\mathbf{z}})^{2n} \right\rangle \;=\; \frac{1}{N_e} \sum_{e} (\mathbf{e}\cdot\hat{\mathbf{z}})^{2n} \;=\; \frac{8 \cdot 0^{n} + 16 \cdot (1/2)^{n}}{24} \;=\; \frac{16}{24} \cdot 2^{-n} \;=\; \frac{2}{3} \cdot 2^{-n}.
$$

Rearranging with $k^{*} = 3$:
$$
\boxed{\left\langle (\mathbf{e}\cdot\hat{\mathbf{z}})^{2n} \right\rangle \;=\; \frac{1}{k^{*}\,2^{\,n-1}} \qquad (n \geq 1,\ \hat{\mathbf{z}}\text{ a principal cubic axis})}
$$

### Consistency check at $n = 1$: the rank-2 isotropic tensor identity

For $n = 1$ the 432 point group forbids any non-isotropic rank-2 invariant, so the moment tensor $\sum_e e_a e_b$ must be proportional to $\delta_{ab}$. Taking the trace, $\sum_e |\mathbf{e}|^2 = N_e$, hence
$$
\sum_e e_a e_b \;=\; \frac{N_e}{d}\,\delta_{ab} \;=\; \frac{24}{3}\,\delta_{ab} \;=\; 8\,\delta_{ab}.
$$
Contracting with $\hat{z}_a\hat{z}_b$ for any unit vector $\hat{\mathbf{z}}$ (not only a principal axis) gives $\sum_e (\mathbf{e}\cdot\hat{\mathbf{z}})^2 = 8$, so $\langle (\mathbf{e}\cdot\hat{\mathbf{z}})^2 \rangle = 1/3$. This recovers the $n = 1$ case and is numerically verified to machine precision by `proofs/cosmology/A_dilution_derivation.py` ("Numerical $\sum_e e_a e_b$ matrix (should be $(24/3)\cdot I = 8\cdot I$)").

### Direction dependence for $n \geq 2$

The formula is **not direction-independent** for $n \geq 2$. For $\hat{\mathbf{z}} = (1,1,1)/\sqrt{3}$ the enumeration of Step 5 is replaced by (undirected) 3 lines with $(\mathbf{e}\cdot\hat{\mathbf{z}})^2 = 2/3$ and 3 lines with $(\mathbf{e}\cdot\hat{\mathbf{z}})^2 = 0$, giving
$$
\left\langle (\mathbf{e}\cdot\hat{\mathbf{z}})^{4} \right\rangle_{(111)} \;=\; \frac{12\cdot(2/3)^2 + 12\cdot 0}{24} \;=\; \frac{2}{9} \;\neq\; \frac{1}{6}.
$$
The principal-axis assumption must therefore be stated and maintained wherever the $n \geq 2$ formula is invoked downstream. The $n = 1$ case is exempt (isotropic rank-2 tensor).

## Result

$$
\left\langle (\mathbf{e}\cdot\hat{\mathbf{z}})^{2n} \right\rangle \;=\; \frac{1}{k^{*}\,2^{\,n-1}} \;=\; \frac{1}{3\cdot 2^{\,n-1}}
\qquad (\hat{\mathbf{z}}\text{ a principal cubic axis}).
$$

Numerical values for $n = 1,\dots,6$:

| $n$ | value |
|-----|-------|
| 1 | $1/3$ |
| 2 | $1/6$ |
| 3 | $1/12$ |
| 4 | $1/24$ |
| 5 | $1/48$ |
| 6 | $1/96$ |

## Comparison with experiment

The srs cubic moment is a graph-intrinsic mathematical identity, not an experimental observable. It enters physics through downstream predictions that contract it against a preferred physical direction; the principal example in this repository is the hemispherical-asymmetry prediction $A = \varepsilon/k = 1/15$ in `predictions/A_hemispherical.py`, which uses only the $n = 1$ case. The $n \geq 2$ cases feed per-$\ell$ Legendre projections discussed in `docs/parameters/parity_theorems.md` §"Per-ell extensions".

## Open questions

1. **Direction restriction.** The formula as stated holds only for $\hat{\mathbf{z}}$ on a principal cubic axis. For the $n \geq 2$ extension to arbitrary directions one needs the full rank-$2n$ moment tensor of the $\langle 110\rangle$ vector set, which has non-trivial cubic anisotropy starting at rank 4. This is a straightforward but separate calculation; no downstream prediction currently uses the $n \geq 2$ formula on a non-principal axis.
2. **Nothing else.** Every step in this derivation is either an upstream `predictions/` file, a cited theorem (Sunada 2012, RCSR), or explicit CAS-checkable arithmetic and enumeration. No fit parameters, no external physics inputs.


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

- O'Keeffe, M., Peskov, M.A., Ramsden, S.J. & Yaghi, O.M. (2008). The Reticular Chemistry Structure Resource (RCSR) database of, and symbols for, crystal nets. *Accts. Chem. Res.* **41**, 1782–1789. [Entry `srs`.]
- Sunada, T. (2012). Lecture on topological crystallography. *Notices AMS* **59**(2), 208–215.
- Wells, A.F. (1977). *Three-Dimensional Nets and Polyhedra*. Wiley-Interscience, New York.
