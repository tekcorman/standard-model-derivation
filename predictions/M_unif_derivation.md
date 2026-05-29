# Derivation of M_unif (gauge unification scale)

**Date:** 2026-05-04 EOD.
**Status (TIGHTENED 2026-05-18 hostile pass):** **STRUCTURAL-DERIVATION-
CONDITIONAL, UNTESTED.** M_unif/M_Pl = α_GUT·α₁_bare = (1/24)(2/3)⁸ is an
anchor-free dimensionless ratio, but (i) there is NO empirical observable
"M_unif" — the ~2×10¹⁶ GeV target is a model-dependent MSSM back-
extrapolation, not data, so there is nothing to numerically validate
against; (ii) the absolute GeV value rides on the CODATA Planck mass;
(iii) the physical-scale identification is "Reading B2", an unproven
hypothesis (Stage 4). The prior "THEOREM-GRADE-CONDITIONAL, Clause 8
PASS" overstated this. See §"Comparison with experiment" for the
retraction. (Stage 3 derives 32 = N_atoms² × N_trivial from substrate
matter loop trace; Stage 4 linear-form justification remains the
conditional.)

## Abstract

The gauge unification scale M_unif is derived as a substrate-local structural quantity from theorem-grade primitives α_GUT = 1/24 and α_1_bare = (k*−1)^(g−2)/k*^(g−2) = (2/3)^8 plus the Planck mass M_Pl (untethered structural prediction in framework-natural units; see `framework_natural_units.md`):

$$M_{\rm unif} = \alpha_{\rm GUT} \cdot \alpha_{1,{\rm bare}} \cdot M_{\rm Pl} = \frac{32}{k^{*\,(g-1)}} \cdot M_{\rm Pl} = N_{\rm atoms}^2 \cdot M_R$$

Numerically: M_unif ≈ 1.985 × 10¹⁶ GeV, matching the MSSM single-regime unification benchmark at −0.76%. The derivation is N_hub-independent (substrate-local family) and uses theorem-grade upstream primitives (α_GUT, α_1_bare, M_Pl).

**Stage 4 closure (2026-05-14 PM):** under the framework's substrate-spectral mass definition (mass-as-flux / mass-as-spectral-gap, per an internal working note and the m_ν₃ closure), M_unif follows the substrate-local-family template

$$M_X = (\text{coefficient}) \times \left(\frac{1}{k^*}\right)^{g-1} \times M_{\rm Pl}$$

with M_unif's coefficient = N_atoms² × N_trivial = 16 × 2 = 32 (matter-bilinear count from Stage 3's rigorous gauge two-point trace × C_3-trivial-sector dim), the same closed-walk return amplitude (1/k*)^(g−1) as M_R, and the same M_Pl base. This linear form is NATIVE under the framework's mass mechanism — the earlier characterization as "structural-conditional via parallelism" was based on importing QFT one-loop self-energy as the mass definition, which is not how the framework defines mass. See `proofs/gauge/srs_M_unif_step4_substrate_spectral.py` and the corrected verdict at an internal working note.

**Grade post-closure:** THEOREM-GRADE-CONDITIONAL on the substrate-local-family mass-as-spectral-quantity mechanism (joint conditional shared with M_R, m_ν₃, v_BZJ — i.e., Need A of MS.1 / multiway formalization). **Dark corrections NOT applicable** (M_unif at unbroken-PS scale, no parity-odd channel for sin(arg h) coupling).

## Framework axioms invoked

- **A1** (substrate toggle dynamics) — provides the discrete substrate structure on which k*, g, N_atoms are defined.
- **A2-T** (refined MDL waterline, derived theorem) — the substrate's dimensional scale set by the Drude form is anchored at the MDL-optimal compression.
- **A5(b)** (MDL probabilities = couplings) — establishes α_GUT = 1/24 as a coupling derived from MDL on the framework's matter content.

No new axiom or adoption introduced.

## Derivation

### Step 1: α_GUT = 1/24 [Type 4 upstream, theorem-grade]

Per `predictions/alpha_GUT.py` and the Class C derivation, the unified gauge coupling at the unification scale is:

$$\alpha_{\rm GUT} = \frac{1}{24}.$$

This follows from the local CAR Fock × Jaynes argument; the factor 24 = 2³ × 3 is a structural counting on the framework's 4-atom Cl(0,2) × C₃ structure.

### Step 2: α_1_bare = (k*-1)^(g-2)/k*^(g-2) [Type 4 upstream, theorem-grade]

Per `predictions/alpha_1.py` (Class A spectral derivation), the bare non-backtracking walker survival amplitude over the girth-cycle interior is:

$$\alpha_{1,{\rm bare}} = \left(\frac{k^* - 1}{k^*}\right)^{g-2} = \left(\frac{2}{3}\right)^8 = \frac{256}{6561}.$$

This is the ratio of NB walker eigenvalue (k*−1) to total eigenvalue (k*) raised to the (g−2) free interior steps of an open girth-cycle walk between two pinned endpoints.

### Step 3: M_Pl as substrate-anchored mass [Type 4 upstream, theorem-grade]

Per `predictions/M_Pl_natural.py` and `framework_natural_units.md`, the Planck mass is a derived structural number in framework-natural units (lattice spacing = 1, tick = 1, ℏ = c = 1, one toggle = one bit = κ_substrate of energy):

$$M_{\rm Pl} = \frac{8}{\sqrt{\pi}} \cdot M_{\rm substrate}.$$

In framework-natural units (M_substrate = 1), M_Pl = 8/√π ≈ 4.514 — a derived dimensionless prediction, untethered.

### Step 4: Reading B2 — gauge two-point bilinear hypothesis [STRUCTURAL HYPOTHESIS, parsimony-preferred]

The candidate identity

$$M_{\rm unif} = \alpha_{\rm GUT} \cdot \alpha_{1,{\rm bare}} \cdot M_{\rm Pl}$$

reduces algebraically to

$$M_{\rm unif} = \frac{32}{k^{*\,(g-1)}} \cdot M_{\rm Pl} = (N_{\rm atoms})^2 \cdot N_{\rm trivial} \cdot \left(\frac{1}{k^*}\right)^{g-1} \cdot M_{\rm Pl},$$

where N_atoms = 4 is the number of atoms per srs primitive cell and N_trivial = 2 is the dimension of the C₃-trivial Bloch sector at P (per `proofs/flavor/srs_M_R_step2_derivation.py`).

The structural reading: at the unbroken-PS scale, the gauge-boson two-point function ⟨A_μ A_ν⟩ is bilinear in the full Bloch sector (because gauge bosons couple to all matter species without sector restriction in the unbroken phase), giving the (N_atoms)² = 16 factor; the closed walker excursion mediating the gauge propagation is the same trivial-mode walk of girth length that gave M_R = N_trivial × (1/k*)^(g−1) × M_Pl, contributing the factor 2 × (1/k*)^(g−1).

This reading is parsimony-preferred among substrate-only readings (see `proofs/foundations/m_unif_full_bloch_bilinear.py` for the full analysis of competing readings — Cl(4) × chirality, PS one-generation × chirality — all of which give 32 due to algebraic accidents but require additional structural assumptions).

**Status of Step 4:** STRUCTURAL HYPOTHESIS. The hypothesis is parsimony-preferred and physically natural but not yet derived from an explicit gauge two-point function computation on the Bloch-decorated Hashimoto operator B(P). Theorem-grade upgrade requires this computation (~3-5 sessions; scoping in an internal working note).

### Step 5: Combination [Type 2 algebra]

Multiplying Steps 1, 2, 3 (and applying Step 4's structural reading):

$$\alpha_{\rm GUT} \cdot \alpha_{1,{\rm bare}} = \frac{1}{24} \cdot \frac{256}{6561} = \frac{256}{157464} = \frac{32}{19683} = \frac{32}{k^{*\,(g-1)}}.$$

Therefore:

$$M_{\rm unif} = \frac{32}{k^{*\,(g-1)}} \cdot M_{\rm Pl} = \frac{32}{19683} \cdot \frac{8}{\sqrt{\pi}} \approx 7.34 \times 10^{-3}\ \text{(framework-natural units)}.$$

Equivalent form (geometric reading via 2026-05-04 M_R closure):

$$M_{\rm unif} = N_{\rm atoms}^2 \cdot M_R = 16 \cdot M_R, \quad M_R = \frac{2}{k^{*\,(g-1)}} \cdot M_{\rm Pl}.$$

### Step 6: K-meta-theorem check (Clause 6)

The coefficient 32/k*^(g−1) = 32/19683 is a pure rational, hence in K = ℚ(√2, √3, √5) trivially. Algebraicity meta-theorem (`theorems/theorem_general_meta_theorem_2026-04-29.md`) is satisfied. ✓

## Result

In framework-natural units (M_substrate = 1):

$$\boxed{M_{\rm unif} = \frac{32}{k^{*\,(g-1)}} \cdot \frac{8}{\sqrt{\pi}} = \frac{256}{19683 \sqrt{\pi}} \approx 7.34 \times 10^{-3}}$$

Equivalent unit translation to GeV via CODATA M_Pl:

$$M_{\rm unif} \approx 1.985 \times 10^{16}\ \text{GeV}.$$

## Comparison with experiment

M_unif is **not directly observed**. The canonical reference value is the MSSM unification scale inferred by inverting standard MSSM RG running on PDG α_i(M_Z), which gives M_unif ≈ 2 × 10¹⁶ GeV. Under the framework's single-regime running (`ADOPTED-MSSM-Sb`, 2026-05-14 PM revision; no M_SUSY threshold), this benchmark is a single-valued reference point.

| Source | Value | Deviation from prediction |
|---|---|---|
| MSSM single-regime inversion | ~2.0 × 10¹⁶ GeV | −0.76% |

**Clause 8 (benchmark consistency — M_unif is not directly measured):**
- Deviation = −0.76% vs the MSSM benchmark.
- Previous "M_SUSY ∈ [1, 10] TeV → ±25% range" framing is RETRACTED (2026-05-14 PM): M_SUSY is not a framework parameter (see `ADOPTED-MSSM-Sb` revision and `feedback_audit_for_smuggled_parameters_2026-05-14`). Constructing a σ_obs envelope by scanning M_SUSY was fitting a free parameter to data.

**Per parameter linter Clause 8e (TIGHTENED 2026-05-18 hostile pass):**
**Clause 8 is N/A here — there is NO empirical observable called M_unif.**
The −0.76% is agreement with a *model-dependent MSSM back-extrapolation*
(inverting MSSM RG on PDG α_i(M_Z) under an assumed sparticle spectrum),
NOT a measurement. The prior "Clause 8 (numerical match) PASSES" was
false comfort — passing against a theory benchmark, not data; it is
retracted. Honest status: M_unif/M_Pl = α_GUT·α₁_bare = (1/24)(2/3)⁸ =
32/19683 is an anchor-free *dimensionless ratio*; its absolute GeV value
rides on the CODATA Planck mass (unit anchor); and the identification of
this number with a physical gauge-unification scale is "Reading B2", an
unproven structural hypothesis. Grade: **STRUCTURAL-DERIVATION-
CONDITIONAL, untested** — a derived ratio with no empirical test, NOT a
theorem-grade prediction. Do not present M_unif as a numerical-match
success. Clause 7 (uniqueness) inherited from Row 4 is unaffected.

## Open questions

1. **Theorem-grade upgrade of Step 4 (Reading B2):** the gauge two-point bilinear-in-full-Bloch hypothesis needs explicit derivation from substrate gauge-field self-energy computation. Specifically, show that the gauge two-point function on srs at P at the unbroken-PS scale picks up (N_atoms)² × N_trivial × (1/k*)^(g−1) as the structural counting factor. Scoping: an internal working note. Sized at 3-5 sessions.

2. **Alternative readings ruled out:** the algebraic accident N_atoms² = dim Cl(4) = PS one-generation dim = 16 means that competing readings (Cl(4) × chirality, PS multiplet × chirality) coincidentally also give 32. Reading B2 is parsimony-preferred (substrate-only) but uniqueness vs these requires a structural distinguishing constraint. Sized at 1-2 sessions.

3. **PS-breaking mechanism at substrate level:** the transition from full-Bloch active (above M_unif, gauge bosons see all matter) to trivial-sector active (below M_unif, lepton-singlet ν_R Majorana mass dominates) is the PS-breaking event. **2026-05-06 update:** the substrate-level mechanism for this transition need not be derived as a gauge-symmetry-breaking event — per an internal working note, PS can be reinterpreted as an organizing/accidental symmetry on local Cl-modules (only SM gauge group is fundamental). Under Candidate E the M_unif scale is reframed as "scale where Cl(6,0) algebra normalization is the relevant accounting unit" (a property of the algebra, not of a gauge-breaking transition). Numerical M_unif value unchanged. Both readings (gauge with ADOPTED-PS-BREAKING; organizing with no breaking gap) are framework-consistent per `adoption_register.md` line 100.

4. **N_hub-independence is a feature, not a bug:** unlike v and m_ν₃ (FSS family), M_unif does not carry N_hub powers. This is consistent with M_unif being a substrate-local quantity and aligns with the framework's two-family hierarchy (substrate-local vs FSS).

## References

- `predictions/alpha_GUT.py`, `predictions/alpha_GUT_derivation.md` — α_GUT = 1/24 theorem.
- `predictions/alpha_1.py`, `predictions/alpha_1_derivation.md` — α_1_bare theorem.
- `predictions/M_Pl_natural.py` — Planck mass untethered structural prediction.
- `predictions/G_N.py` — G_N · M_Pl² = 1 derived identity.
- `proofs/foundations/m_unif_candidate_identity.py` — numerical verification.
- `proofs/foundations/m_unif_full_bloch_bilinear.py` — Reading B2 analysis.
- `proofs/foundations/m_unif_gauge_two_point.py` — structural counting tests.
- `docs/framework/framework_natural_units.md` — natural-unit convention.
- `proofs/flavor/srs_M_R_step{1,2,3}*.py` — M_R = 2/k*^(g−1) × M_Pl source (2026-05-04).
