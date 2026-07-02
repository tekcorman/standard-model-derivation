# Derivation of $\delta_{CP}^{PMNS}$ via the V_{−1}–T_{B-L} Symmetry-Breaking Identity

**Audit anchor:** Row P34, `docs/parameters/parameter_uniqueness_ledger.md`.
**Status:** THEOREM-GRADE-STRUCTURAL — Clauses 1–7 of the parameter-linter rigor gate PASS; Clause 8 PASS at $+0.16\sigma$ vs NuFIT 6.0 IC19 Normal-Ordering best fit. The geometric value is theorem-grade derivable from upstream framework content; one residual adoption (the framework-wide CKM-↔-K_4-walks identification, shared with Row P15 $\delta_{CP}^{CKM}$) blocks an unconditional theorem label.
**File pair:** `predictions/delta_CP_PMNS.py` (this derivation).

---

## 1. Abstract

The Pati–Salam $U(1)_{B-L}$ generator $T_{B-L}$, acting on the 4-vertex $K_4$ atom basis of the framework's substrate at the Bloch $\Gamma$-point, induces a structural symmetry-breaking pattern $SO(3)_{K_4} \to SO(2)_u$ in the $(-1)$-eigenspace $V_{-1}$. The unique $SO(2)_u$-invariant per-atom phase is the polar angle from the broken-symmetry axis $u$, which by direct computation equals $\arccos\bigl(T_{B-L,i}\bigr)$ at atom $i$. For the lepton atom (Slansky 1981 Table 5: $T_{B-L,\text{lepton}} = -1$) this gives $\delta_{CP}^{PMNS} = \arccos(-1) = \pi = 180°$ exactly. The result lies $+0.16\sigma$ above NuFIT 6.0 IC19 Normal-Ordering best fit $177°^{+19}_{-20}$. The derivation requires no fitted parameters; the residual conditional is the framework's existing identification of the $W$-vertex 4-walk Jarlskog phase on $K_4$ with the per-atom polar angle, which is shared with the Row P15 $\delta_{CP}^{CKM}$ closure.

---

## 2. Framework axioms invoked

The derivation uses three foundational axioms of `docs/framework/framework_axioms.md` together with two structural theorems from `docs/theorems/`:

- **A1** — MDL self-containment (forces $d = 3$, $k_* = 3$, srs lattice; cited via `predictions/d_spatial.py` and `predictions/k_star.py`).
- **A2-T** — plural retention with chirality doubling (used implicitly via the framework's Bloch decomposition; cited via `predictions/srs_bloch_dispersion_gamma.py`).
- **A3-T** — substrate Hilbert space is complex (used to define the $V_{-1}$ eigenspace as a real subspace of a real Hermitian operator; cited via `theorem_A3_complex_hilbert_from_multiway.md`).
- **B3** — chirality / spinor-fermion bridge (PS sector content; cited via `predictions/theorem_B3_spinor_fermion.py`).
- **B6** — color-Z_3 multiplicity (gives the 3 color atoms with $T_{B-L} = +1/3$; cited via `predictions/sin2_theta_W.py` Section §4–§5).

Two cited mathematical theorems:

- **Coxeter (1973), *Regular Polytopes*, §7.2** — vertices of the regular tetrahedron inscribed in the unit sphere of $\mathbb{R}^3$ have pairwise inner product $-1/3$; the dihedral angle of the regular tetrahedron is $\arccos(1/3)$.
- **Slansky (1981), *Phys. Rep.* 79, §4 Table 5** — Killing-form-normalized $U(1)_{B-L}$ generator on Pati–Salam $\mathbf{4}$ has eigenvalue $-1$ on the lepton row and $+1/3$ on each of the three color rows.

---

## 3. Derivation

### Step 3.1. The $K_4$ adjacency at the Bloch $\Gamma$-point

By A1, the framework's substrate is the srs lattice ($d = 3$, $k_* = 3$, primitive cell with $k_* + 1 = 4$ atoms). Its Bloch adjacency at the $\Gamma$-point ($k = 0$) is the complete-graph adjacency

$$
A(\Gamma) = J - I,
$$

where $J$ is the all-ones matrix and $I$ is the identity on $\mathbb{R}^4$. This identity is verified by symbolic computation in `predictions/srs_bloch_dispersion_gamma.py` (step 3 asserts $\operatorname{simplify}(A_\Gamma - (J - I)) = 0$).

### Step 3.2. The $(-1)$-eigenspace $V_{-1}$ has tetrahedral structure

The spectrum of $K_4 = J - I$ is $\{+3 \text{ (mult. 1)}, -1 \text{ (mult. 3)}\}$. The $(+3)$-eigenvector is the Perron direction $v_0 = \tfrac{1}{2}(1, 1, 1, 1)$. The $(-1)$-eigenspace is

$$
V_{-1} = \{v \in \mathbb{R}^4 : \langle v, v_0 \rangle = 0\},
$$

a 3-dimensional real subspace. Projecting the four canonical basis vectors $\{e_i\}_{i=0}^{3}$ onto $V_{-1}$ via $q_i := e_i - \tfrac{1}{4}\mathbf{1}$ yields four vectors with

$$
\|q_i\|^2 = \tfrac{3}{4}, \qquad \langle q_i, q_j \rangle = -\tfrac{1}{4} \quad (i \ne j),
$$

so the normalized vectors $q_i/\|q_i\|$ have pairwise inner product $-1/3$. By Coxeter (1973) §7.2, these are the four vertices of a regular tetrahedron inscribed in the unit 2-sphere of $V_{-1}$. The full geometric verification is in `predictions/delta_CP_CKM_geometry.py` and `predictions/delta_CP_CKM_geometry_derivation.md`.

### Step 3.3. The Pati–Salam $T_{B-L}$ acts on $K_4$ atoms

Under the framework's PS sector assignment (B3 + B6), the four $K_4$ atoms of the srs primitive cell carry the Pati–Salam $\mathbf{4}$ representation: one lepton row plus three color rows. The Killing-form-normalized $U(1)_{B-L}$ generator $T_{B-L}$ has diagonal action

$$
T_{B-L} = \operatorname{diag}\bigl(-1,\, +\tfrac{1}{3},\, +\tfrac{1}{3},\, +\tfrac{1}{3}\bigr)
$$

per Slansky (1981) Table 5; this is exactly the eigenvalue assignment used in `predictions/sin2_theta_W.py` `_enumerate_ps_generation` (leptons: $(B-L) = -1$; quarks: $(B-L) = +1/3$).

Since $\operatorname{Tr}(T_{B-L}) = -1 + 3 \cdot \tfrac{1}{3} = 0$, the vector $T_{B-L}\, v_0$ is orthogonal to $v_0$ and hence lies in $V_{-1}$.

### Step 3.4. The symmetry-breaking axis $u$ is exactly anti-parallel to the lepton atom

Define

$$
u := \frac{T_{B-L}\, v_0}{\|T_{B-L}\, v_0\|} \in V_{-1}.
$$

Direct computation:

$$
T_{B-L}\, v_0 = \tfrac{1}{2}\bigl(-1, +\tfrac{1}{3}, +\tfrac{1}{3}, +\tfrac{1}{3}\bigr), \qquad \|T_{B-L}\, v_0\|^2 = \tfrac{1}{4}\Bigl(1 + 3 \cdot \tfrac{1}{9}\Bigr) = \tfrac{1}{3}.
$$

Hence

$$
u = \tfrac{\sqrt{3}}{2}\bigl(-1, +\tfrac{1}{3}, +\tfrac{1}{3}, +\tfrac{1}{3}\bigr) = -\frac{q_{\text{lepton}}}{\|q_{\text{lepton}}\|},
$$

where $q_{\text{lepton}} = q_0 = e_0 - \tfrac{1}{4}\mathbf{1}$. This identity is verified at machine precision in `proofs/foundations/sector_V_minus_one_T_BL_symmetry_breaking_bridge.py` Step 1.

### Step 3.5. Symmetry breaking: $SO(3)_{K_4} \to SO(2)_u$

The regular-tetrahedron point group preserves the $\{q_i\}$ as a set; the inscribed-sphere version is the rotation subgroup $S_4 \cong A_4 \cup \text{(reflections)}$ (Coxeter 1973). $T_{B-L}$ distinguishes the lepton atom (eigenvalue $-1$) from the three color atoms (eigenvalue $+1/3$ each). The residual symmetry under $T_{B-L}$ is the subgroup that permutes the three color atoms while fixing the lepton — this is the cyclic $C_3 \subset S_3 \subset S_4$ acting around the $u$-axis.

In the linear-algebra sense, the residual continuous symmetry group is $SO(2)_u$: rotations of $V_{-1}$ around the axis $u$. The 3 color $q_i$ are mapped to one another (they sit at the same polar angle from $u$, related by $C_3$ azimuthal rotations); the lepton $q_0$ is fixed at the south pole of $u$ (anti-parallel).

### Step 3.6. The unique $SO(2)_u$-invariant per-atom phase

Under $SO(2)_u$, every vector $q \in V_{-1}$ decomposes into

- a **polar angle** $\theta = \arccos\bigl(\langle q, u \rangle / (\|q\|\,\|u\|)\bigr)$ from the $u$-axis (invariant), and
- an **azimuthal angle** $\varphi$ around $u$ (not invariant — picks up a uniform shift under $SO(2)_u$).

The polar angle is the **unique $SO(2)_u$-invariant per-atom phase** (up to an irrelevant sign convention for $u$ vs $-u$, fixed by Step 3.4 to be the lepton-anti-parallel direction).

Compute explicitly using the formulas of Step 3.4:

$$
\cos \theta_i = \frac{\langle q_i, u \rangle}{\|q_i\|\,\|u\|} = \frac{T_{B-L,i}/\sqrt{n}}{\sqrt{(n-1)/n}\,\sqrt{n/(n-1)}/\sqrt{n}} = T_{B-L,i},
$$

with $n = k_* + 1 = 4$. (The cancellation comes from $\sum_i T_{B-L,i}^2 = 1 + 3 \cdot \tfrac{1}{9} = \tfrac{4}{3} = n/(n-1)$, which makes the normalizations cancel cleanly.)

So at every $K_4$ atom $i$,

$$
\boxed{\theta_i = \arccos(T_{B-L,i})}.
$$

For the lepton atom: $\theta_{\text{lepton}} = \arccos(-1) = \pi = 180°$.
For the three color atoms: $\theta_{\text{color}} = \arccos(+1/3) = 70.5288°$.

### Step 3.7. Identification with $\delta_{CP}^{PMNS}$

The framework's existing CKM identification (`predictions/delta_CP_CKM_geometry_derivation.md` Section 6) maps the gauge-invariant $W$-vertex 4-walk Jarlskog phase on $K_4$ to the geometric per-atom angle in $V_{-1}$. The V_{−1}–T_{B-L} reading is the natural unified extension: for an $SU(2)_L$ doublet living at $K_4$ atom $i$,

$$
\delta_{CP}^{(i)} = \theta_i = \arccos(T_{B-L,i}).
$$

For the **color sector** ($i \in \{\text{color}_1, \text{color}_2, \text{color}_3\}$), this gives $\delta_{CP}^{CKM} = \arccos(1/3) = 70.5288°$, recovering Row P15.

For the **lepton sector** ($i = \text{lepton}$), this gives

$$
\delta_{CP}^{PMNS} = \arccos(T_{B-L,\text{lepton}}) = \arccos(-1) = \pi = 180°.
$$

This identification is the residual adoption (Other-Smuggle), discussed in §6.

---

## 4. Result

The closed-form prediction is

$$
\boxed{\delta_{CP}^{PMNS} = \arccos\bigl(T_{B-L,\text{lepton}}\bigr) = \arccos(-1) = \pi = 180.0°.}
$$

Pure-function evaluation in `predictions/delta_CP_PMNS.py` confirms:

```
predict_delta_CP_PMNS(k_star=3, T_BL_per_atom=(-1, 1/3, 1/3, 1/3))
  = 180.000000 deg.
```

Cross-check: substituting a color atom in the same formula reproduces $\delta_{CP}^{CKM} = \arccos(1/3) = 70.5288°$, in agreement with `predictions/delta_CP_CKM_geometry.py` (Row P15 closure) at machine precision.

---

## 5. Comparison with experiment

**Anchor (Normal Ordering best fit):** NuFit-6.0 (Esteban et al., JHEP 12 (2024) 216, arXiv:2410.05380), Table 1, IC19 (without SK atmospheric data):

$$
\delta_{CP}^{PMNS,\,\text{obs}} = 177°{}^{+19}_{-20}.
$$

| Quantity | Value |
|---|---|
| Predicted | $180.000°$ |
| Observed | $177°^{+19}_{-20}$ |
| Absolute deviation | $+3.00°$ |
| Asymmetric $\sigma$ used | $+19$ (upper, since $\Delta > 0$) |
| Deviation in $\sigma$ | $+0.158\sigma$ |
| Clause 8 verdict | **PASS** |

**Cross-check anchor (with SK):** NuFit-6.0 IC24 NO best fit $212°^{+26}_{-41}$; the framework prediction lies $-32°$ from this central value, $-0.78\sigma$ on the lower asymmetric error. Also a Clause 8 PASS, with the **opposite sign** of the deviation. The 35° internal spread between IC19 and IC24 NuFIT 6.0 analyses dominates the framework-vs-observation tension by an order of magnitude.

**Systematic floor:** $\sigma_{\text{theory}} = 0$. $\delta_{CP}^{PMNS}$ is a "pure" structural prediction per Clause 8b — it is not Yukawa-derived, not 1-loop Higgs-sector-derived, and not a quantity requiring SM RG running. The observation precision floor is the only uncertainty source.

**Comparison with retired prediction:** The previous (retired 2026-05-02) formula $\delta_{CP}^{PMNS} = (g_{\text{girth}} - 1) \cdot \arg(h^*) \bmod 360° \approx 249.85°$ had a $+72.85°$ absolute deviation, $+3.83\sigma$ tension under NuFIT 6.0 IC19. Four post-B6 structural routes for $n = g_{\text{girth}} - 1$ surveyed in an internal working note were all NEGATIVE; the V_{−1}–T_{B-L} reading supersedes that approach.

---

## 6. Open questions

### 6.1. The CKM-↔-K_4-walks identification (Other-Smuggle, shared with Row P15)

Step 3.7 identifies the geometric polar angle $\theta_i = \arccos(T_{B-L,i})$ with the gauge-invariant Jarlskog phase $\delta_{CP}^{(i)}$ of the $SU(2)_L$ doublet at $K_4$ atom $i$. This is the framework's existing CKM-↔-K_4-walks identification (per `delta_CP_CKM_geometry §6`), an Other-Smuggle adoption shared with Row P15.

**Status update (2026-05-09):** The earlier framing of this gating cited "Need-A2 + Need-D" as the joint closure dependency. Need-A2 (substrate generation-Z_3 existence) was **CLOSED 2026-05-08** (commit `42a6928`) via the M1.B Galois-tower chain (rediscovered) plus the M_gen non-degeneracy generic measure-theoretic argument. The remaining gate is **Need-D-3 alone** — the $Y_u$ vs $Y_d$ eigenbasis structure on $C^3_{\text{gen}}$.

The 2026-05-05 EOD+3 audit had bounded Need-D-3 closure on Need-A2 alone via Route 4 (SU(2)_L Higgs partner mechanism + chirality-doubled $G2$-D formalization). With Need-A2 now closed, today's (2026-05-09) Need-D-3 attack via the single-σ Galois Z_3 obstruction route was an **HONEST NEGATIVE** — the naive single-σ closure is ruled out at >100σ. Need-D-3 remains BLOCKED on the multi-session M1/M2 substrate-mass-eigenstate program (3–5 sessions per `ckm_substrate_identification_2026-04-29.md`).

Closing Need-D-3 would graduate **both** Row P15 ($\delta_{CP}^{CKM}$) and Row P34 ($\delta_{CP}^{PMNS}$) to UNIQUE-THEOREM-GRADE simultaneously — a high-leverage single closure.

### 6.2. Sector / generation labeling residue (ADOPTED-B3, non-blocking)

The (Z/2)³ Angle D verdict (commit `e5ef667`, 2026-04-30) verified that the SET of predicted $\delta_{CP}$ values $\{\arccos(1/3), \arccos(-1)\} = \{70.53°, 180°\}$ is invariant under the (Z/2)³ relabeling group. Only the *labeling* — that the $\arccos(-1) = 180°$ element corresponds to the PMNS observable rather than the CKM one — carries data-anchored content.

The three (Z/2)³ generators have the following individual closure status (per 2026-05-05 EOD+3 Angle D residue audit):

- **L↔R chirality**: G2-D mirror is preserved, so does NOT break this generator structurally. Closure pending Higgs sector VEV alignment direction (PS-scale).
- **Y sign / lepton ↔ quark**: Killing form fixes magnitudes but not signs; Slansky 1981 convention adopted.
- **Up↔down within doublet**: Cl(1,1) unique up to overall sign per Lounesto 2001.

Per memory, the common structural prerequisite for all three is **Higgs sector VEV alignment direction** (PS-scale + EW-scale), estimated 9–15+ sessions of research-level work. For the predictive content of this row this residue is non-blocking: predictions are (Z/2)³-invariant.

### 6.3. The 35° NuFIT 6.0 internal analysis spread

The IC19 (without SK atmospheric) and IC24 (with SK atmospheric) NuFIT 6.0 analyses give central values 35° apart (177° vs 212°). The framework prediction of 180° straddles both within $1\sigma_{\text{combined}}$, but with **opposite signs of deviation**. This is not a framework-side issue; it reflects that the NO best-fit value of $\delta_{CP}$ remains experimentally unsettled at the ±1σ level. DUNE and Hyper-Kamiokande are projected to resolve $\delta_{CP}^{PMNS}$ to ~5–10° precision.

### 6.4. Inverted Ordering

NuFIT 6.0 strongly disfavors Inverted Ordering ($\Delta\chi^2 = 6.1$ once SK atmospheric data is included), favoring NO. The framework's m_ν1 → 0 prediction (`predictions/m_nu3.py`, `predictions/m_nu2.py` chain) is consistent with NO. No comparison to IO is meaningful here; the framework predicts NO.

---

## Citations

1. Esteban, Gonzalez-Garcia, Maltoni, Schwetz, Pinheiro, "NuFit-6.0: Updated global analysis of three-flavor neutrino oscillations", *JHEP* **12** (2024) 216, arXiv:2410.05380.
2. Coxeter, *Regular Polytopes*, 3rd ed., Dover (1973), §7.2.
3. Slansky, "Group Theory for Unified Model Building", *Phys. Rep.* **79** (1981) 1, §4 Table 5.
4. Lounesto, *Clifford Algebras and Spinors*, 2nd ed., Cambridge (2001), §16.
5. Pati, Salam, "Lepton Number as the Fourth Color", *Phys. Rev. D* **10** (1974) 275.
