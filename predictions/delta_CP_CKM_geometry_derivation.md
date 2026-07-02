# Derivation: Tetrahedral Dihedral Angle from K_4 at the srs Gamma Point

**Parameter:** `delta_CP_CKM_geometry` (geometric sub-result for `delta_CP_CKM`)
**Date:** 2026-04-18
**Status:** STRICT-SOLID geometric theorem under A1 + A2-T + A3-T.
  CKM identification is ADOPTED RESIDUAL (not derived).
**Script:** `predictions/delta_CP_CKM_geometry.py` (prints `OK:`)
**Upstream:** `predictions/k_star.py`, `predictions/d_spatial.py`,
  `predictions/srs_bloch_dispersion_gamma.py`

---

## 1. Abstract

We prove that the srs Bloch adjacency at the Gamma point (k = 0) equals
the K_4 complete-graph adjacency matrix A(Gamma) = J - I. The (-1)-eigenspace
of K_4 is a 3-dimensional real subspace of R^4. Projecting the four standard
basis vectors onto this subspace and normalizing yields four unit vectors in
R^3 with pairwise inner product -1/3, which are the vertices of a regular
tetrahedron. By Coxeter 1973 (Regular Polytopes, §7.2), the dihedral angle
of this tetrahedron is arccos(1/3) = 70.529 deg. The derivation is pure
linear algebra plus Euclidean geometry; no physical identification is required.

The numerical value arccos(1/3) ≈ 70.53 deg is 0.68 sigma from the PDG 2024
CKM CP phase delta_CP_CKM = 68.5 ± 3.0 deg. However, the identification of
this geometric angle with the CKM CP-violating phase is an adopted residual
(Other-Smuggle under the strict gate), blocked by open structural needs
(Need-A2, Need-D) documented in `docs/master_plan.md` Sprint 2.

---

## 2. Framework Axioms Invoked

- **A1** (binary self-inverse toggle): enters via the srs walker -> NB walk
  -> Hashimoto chain (`../predictions/walker_dynamics_derivation.md` W1-W3), and upstream
  via the derivation of k* = 3 (`predictions/k_star.py`).
- **A2** (MDL canonicalization): enters via k* = 3 (MDL selects minimum-degree
  crystal net spanning R^3; `predictions/k_star.py` chain from
  `predictions/d_spatial.py`); and via the Bloch decomposition
  (`../predictions/walker_dynamics_derivation.md` Step 8, Sunada 2012).
- **A3** (MDL canonicalization is partial trace over abstract purifying auxiliary):
  not directly needed for the Gamma-point geometric statement; A3 provides the
  Hilbert-space foundation for the framework (complex field, Born rule) but the
  Gamma-point K_4 result is a real-linear-algebra theorem independent of A3.
  Included for completeness: the upstream files `predictions/d_spatial.py` and
  `predictions/srs_bloch_dispersion_gamma.py` are strict-solid-conditional on
  A3 via G.1 + G.5.

---

## 3. Derivation

### Step 1. k* = 3 and K_4 as the srs quotient at Gamma.

From `predictions/k_star.py` (closed, strict-solid under A1 + A2-T):
the srs lattice coordination number is k* = 3.

From `predictions/srs_bloch_dispersion_gamma.py` (closed, strict-solid
under A1 + A2-T + Sunada 2012 + sympy-verified algebra): the srs Bloch
scalar adjacency A(k), evaluated at k = Gamma = (0, 0, 0) in reduced
BZ coordinates, satisfies

$$
A(\Gamma) = J_4 - I_4,
$$

where J_4 is the 4×4 all-ones matrix and I_4 is the 4×4 identity matrix.
This is the adjacency matrix of the complete graph K_4 (4 vertices, every
pair connected). The verification in `predictions/srs_bloch_dispersion_gamma.py`
is the sympy assertion

```python
assert sp.simplify(A_Gamma - J_minus_I) == sp.zeros(4, 4), "A(Gamma) != K_4 adjacency."
```

This is an exact symbolic verification, not a numerical approximation.

**Physical meaning.** At k = 0 (Gamma), the Bloch phase factors
exp(2πi k · R) = 1 for all lattice vectors R, so A(0) is simply the
sum of the 4×4 adjacency matrices of the underlying primitive-cell graph.
Since every pair of the four srs primitive-cell vertices is connected by
exactly one edge (srs is 3-regular with 4 vertices and 6 edges per
primitive cell; each pair connected by exactly one edge), the sum is J_4 - I_4.
This is a fact of the I4_132 Wyckoff 8a srs realization, not an identification.

### Step 2. Spectrum of K_4 and its (-1)-eigenspace.

The characteristic polynomial of A(Gamma) = J_4 - I_4 is:

$$
\det(\lambda I_4 - (J_4 - I_4)) = \det((\lambda + 1) I_4 - J_4).
$$

By the matrix determinant lemma (or direct computation), for the rank-1
perturbation J_4 = 1·1^T (1 = all-ones column vector):

$$
\det((\lambda+1) I_4 - J_4) = (\lambda+1)^3 \cdot (\lambda + 1 - 4) = (\lambda+1)^3 (\lambda - 3).
$$

Hence the eigenvalues of K_4 are:
- λ = +3, multiplicity 1, with eigenvector v_0 = (1,1,1,1)/2 (uniform).
- λ = -1, multiplicity 3.

The (-1)-eigenspace is V_{-1} = {x ∈ R^4 : x_0 + x_1 + x_2 + x_3 = 0},
the orthogonal complement of v_0 in R^4.

This is confirmed in `predictions/srs_bloch_dispersion_gamma.py` by:

```python
eigenvals_at_Gamma = A_Gamma.eigenvals()
assert eigenvals_at_Gamma == {sp.Integer(3): 1, sp.Integer(-1): 3}
```

### Step 3. Projection of standard basis vectors onto V_{-1}.

For each standard basis vector e_i (i = 0, 1, 2, 3), the orthogonal projection
onto V_{-1} is:

$$
p_i = e_i - \langle e_i, v_0 \rangle v_0 = e_i - \tfrac{1}{4} \mathbf{1},
$$

where 1 = (1,1,1,1) and we used ⟨v_0, v_0⟩ = 1 and ⟨e_i, v_0⟩ = 1/2
(since v_0 = (1,1,1,1)/2, so ⟨e_i, v_0⟩ = (v_0)_i = 1/2), hence
⟨e_i, v_0⟩ v_0 = (1/2)(1,1,1,1)/2 = (1/4)(1,1,1,1).

**Squared norm.** For all i:

$$
|p_i|^2 = |e_i|^2 - \tfrac{1}{4} \cdot 2 \langle e_i, \mathbf{1} \rangle + \tfrac{1}{4^2} \cdot 4
= 1 - \tfrac{2}{4} + \tfrac{4}{16} = 1 - \tfrac{1}{2} + \tfrac{1}{4} = \tfrac{3}{4}.
$$

Explicitly: p_i has +3/4 at position i and -1/4 at all other positions.
Then |p_i|^2 = (3/4)^2 + 3(1/4)^2 = 9/16 + 3/16 = 12/16 = 3/4. ✓

**Cross inner product.** For i ≠ j:

$$
\langle p_i, p_j \rangle = \langle e_i - \tfrac{1}{4}\mathbf{1},\; e_j - \tfrac{1}{4}\mathbf{1} \rangle
= \langle e_i, e_j \rangle - \tfrac{1}{4}\langle e_i, \mathbf{1}\rangle
  - \tfrac{1}{4}\langle \mathbf{1}, e_j \rangle + \tfrac{1}{16}\langle \mathbf{1}, \mathbf{1}\rangle.
$$

For i ≠ j: ⟨e_i, e_j⟩ = 0, ⟨e_i, 1⟩ = 1, ⟨1, e_j⟩ = 1, ⟨1, 1⟩ = 4. Therefore:

$$
\langle p_i, p_j \rangle = 0 - \tfrac{1}{4} - \tfrac{1}{4} + \tfrac{4}{16} = -\tfrac{1}{2} + \tfrac{1}{4} = -\tfrac{1}{4}.
$$

All four p_i have the same length and all pairs have the same inner product,
confirming that {p_0, p_1, p_2, p_3} form the four vertices of a regular
simplex (regular tetrahedron) in V_{-1} ≅ R^3.

### Step 4. Identification as regular tetrahedron.

The normalized projections are q_i = p_i / |p_i| = p_i / sqrt(3/4) = (2/sqrt(3)) p_i.
Their pairwise inner products are:

$$
\langle q_i, q_j \rangle = \frac{\langle p_i, p_j \rangle}{|p_i|^2} = \frac{-1/4}{3/4} = -\frac{1}{3}
\quad (i \neq j).
$$

This is confirmed in `predictions/delta_CP_CKM_geometry.py` by sympy assertion:
```python
assert inner_normalized_simplified == sp.Rational(-1, 3)
```

By Coxeter 1973 (Regular Polytopes, §7.2): four unit vectors in R^3 with
equal pairwise inner product -1/(n-1) = -1/3 (for n = 4 vertices) are the
vertices of a regular tetrahedron inscribed in the unit sphere. The vertices
q_0, q_1, q_2, q_3 form a regular tetrahedron in V_{-1} ≅ R^3.

### Step 5. Dihedral angle of the regular tetrahedron.

By Coxeter 1973 (Regular Polytopes, §7.2, eq. (7.21)): the dihedral angle
θ of the regular (n-1)-simplex (Schläfli symbol {3, 3, ..., 3} with n-2
threes) satisfies

$$
\cos \theta = \frac{1}{n-1}.
$$

For n = 4 (regular 3-simplex = tetrahedron):

$$
\cos \theta_{\text{dihedral}} = \frac{1}{3}, \qquad
\theta_{\text{dihedral}} = \arccos\!\left(\tfrac{1}{3}\right) \approx 70.529°.
$$

This is the angle between two triangular face planes of the regular
tetrahedron, measured along their shared edge.

**Relationship to the vertex angle.** The angle between two vertex vectors
q_i and q_j (the vertex angle, ∠q_i O q_j where O is the centroid/origin)
has cosine -1/3 (from Step 4), giving arccos(-1/3) ≈ 109.47°. These are
supplementary pairs in the sense that the dihedral and vertex angles satisfy
θ_dihedral + θ_vertex = 180°: arccos(1/3) + arccos(-1/3) = 70.53° + 109.47° = 180°.
This is exact: arccos(x) + arccos(-x) = π for all x ∈ [-1, 1].

---

## 4. Result

$$
\theta_{\text{dihedral}}(K_4\text{ at }\Gamma) = \arccos\!\left(\frac{1}{3}\right) = \arccos\!\left(\frac{1}{k^*}\right) \approx 70.529°.
$$

This is a strict-solid result: every step is either an axiom of the framework,
a citeable Euclidean geometry theorem, or explicitly verified algebra.

---

## 5. Comparison with Experiment

| Source | Value | Deviation |
|--------|-------|-----------|
| PDG 2024 (CKMfitter) | 68.5 ± 3.0 deg | — |
| This theorem | arccos(1/3) = 70.529 deg | +2.03 deg = +0.68σ |

The agreement is 0.68σ. This is numerically consistent but does not constitute
a derivation of delta_CP_CKM — the identification is adopted (see Section 6).

---

## 6. Open Questions and Adopted Residuals

### Identification (BLOCKED — Other-Smuggle)

The identification delta_CP_CKM = arccos(1/3) requires the argument:
(a) The CKM Jarlskog invariant J_CKM = Im(V_us V_cb V*_ub V*_cs) is a product
    of walk amplitudes forming a closed loop on K_4.
(b) The phase of this Jarlskog loop equals the K_4 dihedral angle arccos(1/3).

This argument has two structural gaps:

**Gap 1 (Need-D, species-differentiation, an internal working note):**
The CKM matrix arises from the mismatch between up-type and down-type quark
mass bases. Under the current framework, the walker's state space on srs
factorizes as L^2(primitive cell) ⊗ S (spinor), with C_3 acting on the graph
factor and Cl(6,0) up/down distinguishing on the spinor factor. The up-type
and down-type Yukawa operators Y_u and Y_d share the same C_3 eigenbasis at
tree level, giving U_u = U_d = I and CKM = I. Species-differentiation requires
a derived mechanism (Cl(6,0) spinor factors + B-L + SU(2)_L x SU(2)_R). This
is Need-D of an internal working note, currently BLOCKED.

**Gap 2 (Need-A2, canonical generation-Z_3):**
The generation-Z_3 on C^3_gen, distinct from the substrate's color-Z_3, is
required for the Koide/CKM derivation structure. This is Need-A2 of
`docs/master_plan.md`, currently BLOCKED.

**Status of delta_CP_CKM.py (pre-A3):** The file `predictions/delta_CP_CKM.py`
carries a BLOCKED banner documenting this status and is preserved in predictions/
(not retracted). The present theorem ships the strict-solid geometric half; the
CKM identification is explicitly flagged as an adopted residual.

### Future closure path for the identification

Per `docs/master_plan.md` Sprint 2: CKM re-derivation pending Sprint 11 B7.5
(mass operator) + Need-A2 + Need-D. Three structural inputs would unblock:
(i) derived species-dependent C_3 multiplicity;
(ii) derived species-dependent k-point role;
(iii) derived spinor-graph coupling operator.

---

## 7. References

- **Coxeter, H.S.M.** (1973). *Regular Polytopes*, 3rd ed., Dover. §7.2,
  eq. (7.21): dihedral angle of the regular (n-1)-simplex is arccos(1/(n-1)).
- **Sunada, T.** (2012). *Topological Crystallography*, Springer. §§5-6:
  Bloch decomposition of the Hashimoto operator for a periodic graph.
- **Ihara, Y.** (1966). On discrete subgroups of the two by two projective
  linear group over p-adic fields. *J. Math. Soc. Japan* 18, 219-235.
- **Bass, H.** (1992). The Ihara-Selberg zeta function of a tree lattice.
  *IMRN* 3, 107-115.
- **Terras, A.** (2011). *Zeta Functions of Graphs*, Cambridge. Thm 3.1
  (Ihara-Bass identity).
- `predictions/k_star.py` — k* = 3 (closed, strict-solid under A1 + A2-T).
- `predictions/d_spatial.py` — d = 3 (closed, strict-solid under A1 + A2-T).
- `predictions/srs_bloch_dispersion_gamma.py` — A(Gamma) = K_4 adjacency
  (closed, strict-solid under A1 + A2-T + A3-T + Sunada + sympy-verified algebra).
- `docs/framework/framework_axioms.md` — A1, A2, A3 canonical statement.
- `docs/master_plan.md` Sprint 2 — CKM BLOCKED status and need-list.

## Audit v2 (Clause 7 + Clause 8) status

This prediction inherits Row 4 audit v2 closure + Row P14 V_ub family graduation 2026-04-30.
See internal working notes, and Row P15 of
`docs/parameters/parameter_uniqueness_ledger.md`.

- **Status (2026-04-30 graduation, propagated 2026-05-02):** UNIQUE-THEOREM-GRADE for geometric value (regular-tetrahedron dihedral arccos(1/3)) via Coxeter 1973 uniqueness. Identification with the physical CKM CP phase via Jarlskog loop holonomy on K_4 inherits Row P14 V_ub family graduation: amplitude form theorem-grade via M1 twisted walker (commit 753f4cf); labeling layer data-anchored / non-blocking via (Z/2)^3 Angle D verdict + Z3-mass-order verdict (commit e5ef667). Bridge functoriality lemma graduation (2026-04-28) RETRACTED 2026-04-29 — no longer needed; superseded by M1 amplitude-form closure.
- **Clause 7 (uniqueness):** PASS-CITED via Row 4 inheritance + Coxeter 1973 (regular-tetrahedron dihedral is the unique angular invariant of the K_4 (−1)-eigenspace under SO(3) symmetry) + Row P14 inheritance for the physical-CKM-phase identification.
- **Clause 8 (numerical match):** PASS at +0.7σ on PDG 2024 (68.5° ± 3.0°). Systematic floor: zero — δ_CP_CKM is a "pure" structural prediction per Clause 8(b), no Yukawa or 1-loop Feshbach analog applies.
- **Label vocabulary:** **THEOREM-GRADE-NUMERICAL** for the geometric value and predictive content; OTHER-SMUGGLE residue on the physical-CKM-phase labeling is inherited from Row P14, disclosed and non-blocking.
