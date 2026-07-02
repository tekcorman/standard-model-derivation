# Observable-Type Catalogue

**Date:** 2026-04-18
**Status:** working document — entries added as reading-rule status is determined
**Foundation:** `../../predictions/mdl_symmetry_coherence_derivation.md` (MDL Symmetry Coherence theorem, closed)

This catalogue maps each class of Standard Model observable to its topological source in
the srs multiway graph and to the reading-rule case that applies under A1 + A2 + A3. The
MDL Symmetry Coherence theorem (closed) establishes two reading-rule cases:

- **Coherent (Type-C):** paths related by Aut(srs) at a Gamma-fixed k-point; amplitudes sum
  with character phases; p = |Σ chi(g) A(gamma_0)|² / Z
- **Incoherent (Type-I):** paths with distinct reduced-word labels in F_inv(E); probabilities
  multiply; p = ((k-1)/k)^L = (2/3)^L on srs

Every new observable fits one of these two cases, or requires a new topological structure
that extends the catalogue. **Default is BLOCKED** (rigor bar: an internal note).

---

## 1. Coupling strengths (Feshbach type)

**Reading-rule case:** Type-I (incoherent product)

**Topological source:** NB girth cycles on srs with n_fixed pinned edge positions.
- Girth g = 10 (closed, `predictions/g_girth.py`)
- k* = 3 (closed, `predictions/k_star.py`)
- Coupling = ((k-1)/k)^(g - n_fixed) = (2/3)^(g - n_fixed) for n_fixed in {0, 1, 2}

**Why incoherent:** Sequential steps in a girth cycle are distinct reduced words in F_inv(E)
(no automorphism relates step j to step j+1); they are MDL-distinguishable; product rule
applies (MDL Symmetry Coherence theorem, Part b, Step 5).

**Derivation status:** STRICT-SOLID (strict core) + I-Feshbach (physical identification of NB
survival rate with scattering coupling is a separately load-bearing commitment, not yet
derived from A1+A2+A3 alone — flagged in `predictions/feshbach_exponent_principle.py`).

**Instantiated predictions (closed at strict-solid level):**
- `predictions/feshbach_exponent_principle.py` — n_fixed in {0, 1, 2}; coupling values 0.026,
  0.039, 0.059

**What closes I-Feshbach:** Green's function derivation identifying NB survival probability
with the physical scattering amplitude (propagator pole residue). Not yet attempted.

---

## 2. Spectral ratios (Koide type)

**Reading-rule case:** Type-C (coherent sum at a C_3-fixed k-point)

**Topological source:** C_3-isotypic decomposition of the Ramanujan subspace V_Ram at the
P-point of srs.
- P-point is C_3-fixed (closed, `../../predictions/B_P_doubly_degenerate_h_derivation.md`)
- Ramanujan subspace is 8-dimensional with C_3 isotypic multiplicities (4, 2, 2) for
  irreps (trivial, omega, omega-bar) (closed, `docs/theorem_B5_3_core.md`)

**Why coherent:** C_3-related Bloch fibre paths at a C_3-fixed k-point are MDL-indistinguishable
(same graph-theoretic properties, same spectral content); equal magnitudes are forced by
A2 + Grunwald 2007 Sections 5.1-5.3; phases are chi(g) = omega^j by Serre 1977 Section 2.3
(MDL Symmetry Coherence theorem, Part a, Steps 1-3).

**Amplitude formula (closed):**

    amp_j = sqrt(mu_trivial) + sqrt(mu_omega) omega^j + sqrt(mu_omega_bar) omega^(-j)
           = sqrt(4) + sqrt(2) omega^j + sqrt(2) omega^(-j)

where j in {0, 1, 2} is the C_3 irrep index (= generation index under ADOPTED-Z3).

**Derivation status of the spectral arithmetic:** STRICT-SOLID (color-sector Born-rule
identity, closed from A1+A2+A3+CDT 2011+Serre 1977+Grunwald 2007+upstream files).

**Residual identifications (separately load-bearing, not yet derived):**
- ADOPTED-P1: amplitudes supported on V_Ram (not V_tree or full fiber)
- ADOPTED-Y: substrate amplitude = Yukawa coupling amplitude (not Feshbach survival, not flux)
- ADOPTED-Z3: irrep index j = generation label (electron/muon/tau or up/charm/top)

**Instantiated predictions:**
- Q_Koide = 2/3 (exact) — `predictions/Q_Koide.py`
- epsilon_Koide = sqrt(2) (exact) — `predictions/epsilon_Koide.py`
- delta_Koide = 2/9 (exact) — `predictions/delta_Koide.py`

**What closes ADOPTED-P1:** Feshbach projection route (A2 MDL comparison between V_Ram and
V_tree descriptions); ADC gap is the current obstruction. See `../audits/registers/adoption_register.md`.

**What closes ADOPTED-Z3:** No working route; cocycle obstruction proven for A_4 route;
Pati-Salam Cartan incompatibility proven for B4 route. Two untried routes remain (Yukawa
Z_3 and Higgs Bloch Z_3), both contingent on Sprint 7+.

---

## 3. Algebraic combinations (derived, no new topological input)

**Reading-rule case:** Closed algebra over Type-C entries; no separate topological source.

**Rule:** Any rational or algebraic function of Type-C amplitudes that is computable from
the (4, 2, 2) multiplicities alone is STRICT-SOLID with the same adoption structure as
the Type-C inputs it depends on.

**Instantiated predictions:**
- delta_Koide = Q * (1 - Q) = 2/9: purely algebraic from Q_Koide
- epsilon^2 = 6Q - 2 = 2: Bernoulli moment identity (cross-check for epsilon_Koide)

**Scope:** Limited to functions of {mu_trivial, mu_omega, mu_omega_bar} = {4, 2, 2} and
the C_3 character values {1, omega, omega^2}. Anything requiring a new topological input
(different k-point, different subspace, different group) is a new catalogue entry.

---

## 4. Mass scale (T_mass — OPEN)

**Reading-rule case:** UNKNOWN — this is the central open problem.

**Topological source:** Not determined. Three candidates identified and assessed:

**Candidate 4a: Bloch dispersion at a high-symmetry k-point (Sprint 7 BLOCKED)**
- Conjecture: Higgs VEV = order parameter at a Curie-Weiss critical point of the srs
  spectral order (P-point, B(P) leading eigenvalue h).
- Assessment: Curie-Weiss FSS attempt done;
  BLOCKED at three points: (F1) Koide delta prefactor not derived; (F2) Higgs identification
  requires SU(2)_L on vertex space but SU(2)_L lives in the spinor (edge) sector; (F3)
  H_0 and t_P are external inputs not derivable from A1+A2+A3 alone.
- Numerical check: v_predicted = 249.74 GeV vs v_observed = 246.22 ± 0.26 GeV; 3.5 sigma.
  Not passable as a theorem.

**Candidate 4b: Schmidt coefficients of A3 purification (BLOCKED)**
- Conjecture: mass eigenvalues = Schmidt coefficients of the global A3 purification on
  H_multiway tensor H_aux.
- Assessment: attempted in an internal working note; BLOCKED because the
  Schmidt spectrum is determined by the full H_aux structure, which is abstract and
  operationally under-constrained (Sub-result NEC-Y gives four necessary conditions N1-N4
  for any A3-derived mass reading, none yet sufficient).

**Candidate 4c: density-matrix diagonal of the A3 reduced state (BLOCKED)**
- Conjecture: mass eigenvalues = diagonal of rho_visible = Tr_aux(psi)(psi*) in the
  C_3-isotypic basis.
- Assessment: same blocked status as 4b — the diagonal depends on |psi>_{aux} which is
  abstract.

**Current status:** T_mass is the single largest open question in the framework. ADOPTED-Y
holds the placeholder; no derivation route is currently open.

**What would close T_mass:**
- A derivation of the Higgs VEV from srs spectral data (closes ADOPTED-Y = ADOPTED-F2);
  requires identifying a physical scale in the srs structure without external input.
- OR a derivation of the A3 Schmidt spectrum from A1+A2+A3 alone (closes the operational
  under-determination of H_aux).
- OR a new topological feature of the srs multiway graph that serves as a mass operator
  (proposed but not yet instantiated).

---

## 5. Mixing angles (T_mixing — OPEN)

**Reading-rule case:** Partial — candidate structure identified, not yet derived.

**Topological source (candidate):** Inter-band matrix elements of B(k) between C_3-isotypic
sub-bundles at k-points connected by a BZ path.

**Why candidate:** Mixing angles measure the overlap between mass eigenstates (generation
basis = C_3 irrep basis at P) and weak-interaction eigenstates (flavor basis = some other
k-point or operator basis). The overlap is an inner product between Bloch states at
different k-points or in different isotypic components, which is naturally an inter-band
matrix element of the Bloch operator B(k).

**What is known:**
- The C_3 isotypic decomposition (4,2,2) gives the mass-eigenstate basis (Type-C, closed).
- The weak-interaction eigenstate basis requires identifying a second basis in H_P or at a
  different k-point where the gauge interaction is diagonal.
- The two bases do not coincide if and only if the Bloch bundle is non-trivially twisted
  between the mass-eigenstate and flavor-eigenstate k-points.

**Current status of specific mixing angles:**
- V_us, V_cb, V_ub: Feshbach-pattern predictions shipped (`predictions/V_us.py`,
  `predictions/V_cb.py`, `predictions/V_ub.py`) with ADOPTED-PS flagged; numerical agreement
  is within experimental errors but derivation of the topological source is not yet closed.
- PMNS theta_12, theta_23, theta_13: pre-A3 derivations retracted; reading rule for phases
  requires Need-RR closure (blocked on T_mass first).
- Delta_CP (PMNS CP phase): requires arg(h) = physical CP phase; blocked on both Need-RR
  and T_mixing.

**What would close T_mixing:**
- Identify the flavor basis as a specific other Bloch fibre (e.g., Gamma-point, H-point,
  or N-point) from A2 MDL selection of the "weak-interaction-diagonal" k-point.
- Derive the inter-band overlap formula from the Bloch bundle connection (Berry connection
  between C_3-isotypic sub-bundles along a BZ path).
- This requires closing T_mass first (need the mass-eigenstate basis to be derived before
  computing overlaps).

---

## 6. Neutrino masses and mass-squared splittings

**Reading-rule case:** Hybrid — strict core is Type-I (Feshbach Exponent Principle); mass
scale is Type T_mass (open, ADOPTED-PS placeholder).

**Topological source (strict core):** Feshbach correction factor from the P-point shape
parameter Im(h)/|h|^2 = sqrt(5)/4.
- h = (sqrt(3) + i*sqrt(5))/2, |h|^2 = 2 (closed, `../../predictions/B_P_doubly_degenerate_h_derivation.md`)
- Shape factor: Im(h)/|h|^2 = (sqrt(5)/2) / 2 = sqrt(5)/4 (strict-solid)

**Mass-squared ratio R = 228/7:**
- Closed, `predictions/R_nu_splitting.py`; algebraic from srs spectral data

**Predictions shipped:**
- m_nu2 = 8.6436 meV; deviation -0.10 sigma — `predictions/m_nu2.py`
- m_nu3 = 49.3300 meV; deviation -4.01 sigma — `predictions/m_nu3.py`

**What the -4.01 sigma means:** The theorem-grade Feshbach correction is sound. The deviation
is driven entirely by the A-grade external input (ADOPTED-PS: m_nu3_bare from Pati-Salam RG
pipeline). The strict-solid content is the correction factor 1 + (sqrt(5)/4) * alpha_1_bare;
the scale is not derived.

**What closes the mass-scale gap:** Sprint 10 (G_Newton from A1+A2+A3 → M_Planck → M_GUT →
M_seesaw → m_nu3_bare); entirely contingent on closing T_mass first.

---

## 7. Fine-structure constant (alpha_em)

**Reading-rule case:** TBD under A3 — pre-A3 derivation exists but has not been re-derived.

**Topological source (pre-A3 claim, not yet audited under A3):**
- Pre-A3 derivation: alpha_em from NB walk self-energy on srs girth cycle; numerical value
  obtained. Not in predictions/ under A3 rigor.

**Current status:** Listed as a Sprint 8 or later parameter. No attempt under A1+A2+A3 yet.
No post-A3 file exists.

**What closes it:** Re-derive from A1+A2+A3 under the Feshbach pattern. If the pre-A3
derivation used the girth cycle, the Type-I (incoherent) case of the MDL Symmetry Coherence
theorem is the natural framework. Need to identify which n_fixed gives the EM coupling scale.

---

## 8. Electroweak mixing (sin²θ_W)

**Reading-rule case:** UNKNOWN

**Topological source:** Not identified. The Weinberg angle is a ratio of gauge couplings
g'/sqrt(g^2 + g'^2); in the Pati-Salam model it is a group-theoretic ratio at unification.
No srs spectral feature has been mapped to this ratio.

**Current status:** BLOCKED — no attempt made, no candidate topological structure identified.

**What would open this:** Identifying the SU(2)_L × U(1)_Y gauge structure as a feature of
the srs spinor (edge-space) sector derived from A1+A2+A3, then computing the ratio of the
corresponding coupling strengths via Feshbach or Type-C formula.

---

## 9. Strong coupling (alpha_s)

**Reading-rule case:** UNKNOWN — same structural gap as sin²θ_W.

**Current status:** BLOCKED, no attempt.

---

## 10. Higgs quartic coupling (lambda_Higgs)

**Reading-rule case:** Type-I (incoherent) — Feshbach pattern, same as coupling strengths;
plus a dark-correction class factor (tan²(arg h)) from the P-point eigenvalue structure.

**Topological source:**
- NB walk survival on universal covering tree: (k-1)/k)^(g-2) = (2/3)^8 = α₁_bare (strict-solid)
- Dark-correction class factor: tan²(arg h) = Im(h)²/Re(h)² = 5/3 (strict-solid algebra)
- Combined: α₁_full = (5/3)×(2/3)^8
- Factor 2 multiplier: number of Cl(2) generators = number of complex edge boolean DOF

**Derivation status (2026-04-18):** ADVANCED — three adopted steps remain open:
1. **I-Feshbach** — identifying α₁_bare with the physical Feshbach coupling magnitude; requires
   completing the 12×12 K₄-quotient matrix calculation (finite, unwritten).
2. **Dark-map Class 2** — classifying λ as the "mass²-class" (coefficient 5/3) from
   `dark_correction_theorem_2026-04-14.md` §4a; adopted physical classification, not A1+A2+A3.
3. **F2-class factor 2** — identifying the Cl(2) generator count with the Higgs doublet complex
   dimension; requires Sprint 7a F2 closure (B3 spinor-fermion identification, not yet dispatched).

**Prediction file:** `predictions/lambda_higgs.py` — all three adopted steps explicitly labeled.
Exact rational: λ = 2560/19683 ≈ 0.13006. Match: +0.52% (+1.7σ vs observed 0.1294 ± 0.0004).

**Residual identifications (separately load-bearing):**
- ADOPTED I-Feshbach: K₄ matrix calculation closes this (see `../../predictions/Feshbach_coupling_strength_derivation.md` §9)
- ADOPTED dark-map: requires first-principles classification of observable types
- ADOPTED F2-class: contingent on Sprint 9 B3 + Sprint 7a F2 closure

---

## 11. Fermionic exchange statistics — structural foundation (A4 gap)

**Reading-rule case:** N/A — this is a foundational question, not an observable class.

**The gap:** The Cl(6) ⊗ Cl(2) = Cl(8) algebraic structure underlying fermion mass mixing and
the Standard Model spinor decomposition requires that edge modes at k*-valent graph nodes satisfy
canonical anticommutation relations (CAR). This is NOT derivable from A1+A2+A3:

- A1 supplies toggle involution on edges (self-inverse). No sign structure.
- A2 selects reduced words by description length. Multiway paths differing only in swap order
  at a shared node are MDL-equidistinct (same description length, same graph properties) — no
  sign enforcement emerges.
- A3 encodes pure/mixed structure via partial trace. No grading or parity constraint.

Without fermionic exchange statistics (CAR), the node Fock space is bosonic (Weyl algebra),
Cl(6) collapses to the bosonic oscillator algebra, and Cl(8) = Cl(6) ⊗ Cl(2) cannot be built.

**Proposed A4 (node grading):** "Edge modes at each k*-valent node in srs satisfy canonical
anticommutation relations (Jordan-Wigner CAR), not commutation relations. Equivalently: the
state space at each node is graded by fermionic parity."

**Status:** A4 is a GENUINE FOURTH AXIOM — not reducible to A1+A2+A3. If adopted, it
simultaneously closes:
- Cl(6) fermionicity (3 edge modes → 8-dim fermionic Fock space)
- Cl(2) anticommutativity (2 edge booleans acquire sign from Jordan-Wigner ordering)
- Cl(8) = Cl(6) ⊗ Cl(2) derivation (both factors now have a canonical construction)

**Open question:** What physical principle selects fermionic over bosonic statistics at nodes?
Candidates: (a) spin-statistics from Lorentz invariance (requires spacetime first); (b) MDL
grading of Fock-space descriptions (untested); (c) pure adoption as a structural input.

---

## Summary table

| Observable class | Reading-rule case | Topological source | Adoptions remaining | Prediction files |
|---|---|---|---|---|
| Feshbach couplings | Type-I (incoherent) | NB girth cycles, n_fixed pinned | I-Feshbach | `feshbach_exponent_principle.py` |
| Koide ratios (Q, eps, delta) | Type-C (coherent) | C_3 orbits at P, (4,2,2) mults | ADOPTED-P1, -Y, -Z3 | `Q_Koide.py`, `epsilon_Koide.py`, `delta_Koide.py` |
| Algebraic combinations | Closed algebra | None beyond Type-C inputs | inherited | `delta_Koide.py` (cross-check) |
| Mass scale M | T_mass — OPEN | Not identified | ADOPTED-Y (= ADOPTED-F2) | None shipped (BLOCKED) |
| Mixing angles (CKM, PMNS) | T_mixing — OPEN | Inter-band B(k) overlaps (candidate) | T_mass first | `V_us.py`, `V_cb.py`, `V_ub.py` (Feshbach, ADOPTED-PS) |
| Neutrino masses | Hybrid (I + T_mass) | Feshbach correction strict; scale open | ADOPTED-PS | `m_nu2.py`, `m_nu3.py` |
| alpha_em | TBD | Pre-A3 girth cycle (needs audit) | audit needed | None under A3 |
| sin²θ_W | UNKNOWN | Not identified | — | None |
| alpha_s | UNKNOWN | Not identified | — | None |
| lambda_Higgs | Type-I + dark-map | NB walk (2/3)^8 × tan²(arg h) × 2 | I-Feshbach, dark-map Class 2, F2-class factor | `lambda_higgs.py` (ADVANCED) |
| Fermionic statistics | Foundational (A4 gap) | Not topological — requires CAR postulate | A4 entire (not derivable A1+A2+A3) | None — structural question |

---

## Closure priority order

Given the dependency structure, the blocking chain is:

    T_mass → T_mixing → CKM/PMNS → mass-scale closure of m_nu / lepton masses
    T_mass → ADOPTED-Y → ADOPTED-P1 (Feshbach route) → Q_Koide full identification

**Sprint order implied:**
1. T_mass (any route) — unlocks everything downstream
2. I-Feshbach (Green's function derivation) — unlocks Feshbach couplings as theorem-grade
3. ADOPTED-Z3 (generation labeling) — unlocks Koide identification with charged leptons
4. T_mixing (inter-band overlaps) — unlocks CKM and PMNS
5. alpha_em post-A3 audit — independent, can parallelize with 1-4
