# Derivation: α_GUT (Gauge Coupling at the Unification Scale)

**File:** `predictions/alpha_GUT.py`
**Status:** THEOREM under A1 + A2-T + local CAR thm + A5(b) + Jaynes 1957 + Kraft 1949
**Date:** 2026-04-19 (session 2)

---

## Abstract

We derive the Standard Model gauge coupling at the unification scale, α_GUT, from the local label count at a trivalent node of the srs lattice. Under the local CAR thm (CAR/fermionic statistics), the Fock space at a k*-valent node has dimension 2^k* = 8 for srs. Combined with k* = 3 incident directed edges (A1 + MDL), the local label space has 2^k* × k* = 24 elements. Under A2-T + Jaynes maximum-entropy prior, each local event has probability 1/24. Under A5(b), this MDL probability is identified with the gauge coupling at the unification scale, giving α_GUT = 1/24 ≈ 0.04167. The observed value α_GUT ≈ 1/24.3 (from MSSM RG running of measured M_Z gauge couplings) matches to 1.3%, well within typical threshold-correction uncertainty for a tree-level prediction.

---

## Framework Axioms Invoked

- **A1** (binary self-inverse toggle): supplies the underlying graph structure.
- **A2-T** (MDL canonicalization; derived theorem): selects srs as the unique MDL-minimal lattice and forces uniform prior over indistinguishable local labels. See `docs/theorems/theorem_A2_mdl_from_finite_register.md`.
- **local CAR thm** (node grading / CAR; derived theorem): supplies the Fock-space dimension 2^k* at a k*-valent node. See `docs/theorems/theorem_car_local_jordan_wigner.md`.
- **A5(b)** (coupling clause): identifies MDL leading-order probability with physical coupling strength.

Plus standard mathematical theorems:
- **Jaynes 1957** (maximum entropy): uniform prior on a finite set with no further constraints.
- **Kraft 1949** (description length / inequality): relates label count to bits.

---

## Derivation

### Step 1: Local state space at a trivalent node

By A1 (binary toggle on edges) + MDL selection of srs (`predictions/k_star.py`, STRICT-SOLID under A1 + A2-T + Delgado-Friedrichs & O'Keeffe 2003), each node of the srs lattice has

$$k^* = 3$$

incident directed edges.

By the local CAR thm (CAR / fermionic statistics at trivalent node, `docs/theorems/theorem_car_local_jordan_wigner.md`), the local Fock space at each k*-valent node has dimension

$$\dim \mathcal{F}_{\rm node} = 2^{k^*} = 2^3 = 8.$$

These 8 states are the standard Fock states of three fermionic edge modes:

$$\{|000\rangle, |001\rangle, |010\rangle, |011\rangle, |100\rangle, |101\rangle, |110\rangle, |111\rangle\}$$

physically corresponding to the eight Cl(6) ground states of one Standard Model generation (per the B5.3 core decomposition).

The total **local label space** at a node — combining (Fock state) and (edge direction) — has size

$$N_{\rm local} = (\dim \mathcal{F}_{\rm node}) \times k^* = 2^{k^*} \cdot k^* = 8 \times 3 = 24.$$

### Step 2: Uniform MDL prior

Under A2-T (MDL canonicalization) with no further constraints distinguishing the 24 local labels, the maximum-entropy prior is uniform (Jaynes 1957). The MDL probability of a specific label is therefore

$$P(\text{specific local label}) = \frac{1}{N_{\rm local}} = \frac{1}{24}.$$

By Kraft's inequality (Kraft 1949), the description length of a uniformly distributed label from a 24-element set is

$$DL = \log_2(N_{\rm local}) = \log_2(24) = \log_2(3 \cdot 2^3) = \log_2(3) + 3 \approx 4.585 \text{ bits}.$$

The structural interpretation:
- **log₂(3)** = bits to specify which of k* = 3 edge directions
- **3** = bits to specify which of 2^k* = 8 Fock states (= log₂(2^k*) = k* bits)
- Sum: log₂(3) + 3 = log₂(24) = 4.585 bits.

### Step 3: Physical identification (A5(b))

By A5(b) (coupling clause; `docs/framework/framework_axioms.md` §5b, established 2026-04-19): the MDL probability of a leading-order multiway process is identified with the physical coupling strength of that process.

The leading-order gauge-mediated event at a node is specified by minimal labels (one Fock state, one direction) — i.e., the simplest possible local event. Higher-order events (multiple Fock states, multiple directions, multi-node) carry more bits and contribute at higher orders in the perturbative expansion.

Therefore:

$$\alpha_{\rm GUT} = P(\text{leading-order local event}) = \frac{1}{N_{\rm local}} = \frac{1}{2^{k^*} \cdot k^*} = \frac{1}{24}.$$

This is the **gauge coupling at the unification scale** — the natural energy scale of the framework (the lattice scale, comparable to M_GUT ≈ 2 × 10^16 GeV). At lower energies, RG running breaks the unified coupling into the Standard Model gauge couplings α_1, α_2, α_3 with the dim(SU(3)):dim(SU(2)):dim(U(1)) = 8:3:1 ratio (standard GUT embedding).

---

## Result

$$\boxed{\alpha_{\rm GUT} = \frac{1}{2^{k^*} \cdot k^*} = \frac{1}{24} \approx 0.04167}$$

For srs (k* = 3): α_GUT = 1/24 exactly as a rational number.

---

## Comparison with Experiment

| Quantity | Value |
|----------|-------|
| Predicted α_GUT | 1/24 = 0.04167 |
| Observed α_GUT (MSSM RG running, M_Z anchor) | ≈ 1/24.3 ≈ 0.04115 |
| Deviation (absolute) | +0.00052 |
| Deviation (relative) | +1.3% |

The "observed" value comes from MSSM renormalization-group running of the measured Standard Model gauge couplings g_1, g_2, g_3 at M_Z up to the unification scale M_GUT ≈ 2 × 10^16 GeV. Canonical MSSM literature (Amaldi-de Boer-Fürstenau 1991; Langacker & Polonsky 1995) gives α_GUT^{-1} ≈ 24.3 ± 0.5, depending on the SUSY spectrum and threshold corrections.

The 1.3% discrepancy between 1/24 (predicted, tree-level) and 1/24.3 (observed, RG-corrected) is within the typical range for two-loop and threshold corrections at M_GUT. The framework's prediction is the **bare** coupling at the lattice scale; full agreement with the observed RG-corrected value would require including SUSY threshold effects, which are standard physics not part of this derivation.

---

## Note on the |S₄| / |Aut(K₄)| structural reading (added 2026-04-19; tightened 2026-05-03)

**An alternative structural reading of the same number 24 is available via the quotient-graph automorphism group.** K₄ (the complete graph on 4 vertices = the tetrahedron) is the local incidence model of the srs (Laves) net — the smallest connected 3-regular graph. Its automorphism group is |Aut(K₄)| = |S₄| = 4! = 24, since any vertex permutation preserves the complete-graph edge structure.

Reading B: α_GUT = 1/|Aut(K₄)| = 1/|S₄| = 1/24.

This project's canonical derivation (Reading C, §1–§3 above): N_local = (Fock dim) × (edge directions) = 2^{k*} × k* = 8 × 3 = 24, with α_GUT = 1/N_local = 1/24 by uniform MDL prior + A5(b).

**Two readings, one number. Algebraically equivalent or numerically coincident?**

The numerical agreement 8 × 3 = 4! = |S₄| = 24 is arithmetic, not by itself structural. The two decompositions are physically distinct:
- Reading C decomposes 24 as (Fock state) × (edge direction) at a single srs site. Per-site local label space.
- Reading B reads 24 as |S₄|-permutations of 4 vertices of the K₄ quotient graph. Quotient-level automorphism count.

|Aut(K₄)| = |S₄| acts on 4 vertices of K₄, not on 3 edges + 8 Fock states of one srs site. Without an explicit S₄-equivariant identification (Cl(6) Fock space at one site) × (3 edges) ↔ (S₄ permutations of K₄ vertices), the equality 8 × 3 = |S₄| remains numerical, not algebraic.

**Status as of 2026-05-03:** Two structurally distinct readings, same numerical answer. Whether 24 = 8 × 3 = |S₄| reflects a deeper algebraic identity (e.g., an S₄-equivariant decomposition of the Cl(6) Fock space tensored with the edge labels) is an OPEN structural question. The project's canonical derivation (Reading C) is theorem-grade as it stands; Reading B is documented here as cross-validating the numerical value via an independent structural angle.

**RESOLUTION 2026-05-21 — Reading B RETIRED as a structural claim.** `proofs/foundations/gauge_hub_stage5_structure_group_forcing_2026-05-21.py` settled the open question by exact finite-group computation. The S₄-equivariance test fails: the substrate's natural group on the 24 = (Fock dim 8) × (edge dirs 3) local labels is **G_nat = (Z₂)³ ⋊ Z₃ = Z₂ × A₄** — the 3 edge qubits carry (Z₂)³, the body-diagonal C₃ cycles them — which is **not isomorphic to S₄** (S₄ is centerless with dihedral D₄ Sylow-2; G_nat has center Z₂ and elementary-abelian (Z₂)³ Sylow-2; element-order profiles differ — G_nat has order-6 elements, S₄ has order-4). The (Cl(6) Fock) ⊗ (edge labels) space is *not* an S₄-regular representation. By the criterion stated in the "Recommended scoping" paragraph below — "if it doesn't [hold], the |S₄| identification is honestly numerical and should be retired as a structural claim" — Reading B is hereby **retired**: 24 = 8×3 = |S₄| is a numerical coincidence of two counts of non-isomorphic groups, not an algebraic identity. Reading C (the canonical derivation, §1–§3) never used S₄ and is unaffected. The deeper finding: the reading "1/24 = dim(triv)/|G|" is *group-blind* — it holds for all 15 order-24 groups (the trivial rep is always 1-dimensional) — so it forces only |G| = 24 = N_local, exactly Reading C's input; the group-theoretic reframe reduces no input.

**H¹ Master Theorem cross-check on K₄.** `docs/theorems/theorem_h1_master_compression.md` Theorem 1 applied to K₄ (n = 4, k = 3):
- dim C¹ = kn/2 = 6
- dim B¹ = n − 1 = 3
- dim H¹ = (k − 2)n/2 + 1 = 3

Gauge / physical = 1 (small-n; doesn't reach the asymptotic 2/(k−2) = 2 of an infinite trivalent graph). K₄ is finite and exact — useful as an analytic test bed but does not by itself derive α_GUT from cohomology alone.

**Recommended scoping (not work for this pass).** A separate scoping doc (an internal working note — TODO) could attempt to establish or refute the algebraic equivalence (Cl(6) Fock space) ⊗ (edge labels) ↔ S₄-permutations of K₄ vertices. If equivalence holds, the |S₄| reading would graduate from "cross-check" to "second algebraic derivation." If it doesn't, the |S₄| identification is honestly numerical and should be retired as a structural claim.

**Note (2026-05-03):** the |S₄| reading is documented here as cross-validation of the canonical 1/(2^k* × k*) derivation. Until the S₄-equivariance question is settled, the project's load-bearing derivation remains §1–§3 above; the |S₄| match is a structurally distinct route arriving at the same number.

## Open Questions

1. **"Leading-order" specificity in A5(b).** A5(b) identifies leading-order MDL probabilities with coupling strengths. For α_GUT we identify "leading-order" with the minimal-DL event (one Fock state, one direction). This is structurally natural but worth a separate formal justification — analogous to the leading-order Lemma G1 identification used for α_1.

2. **Decomposition into α_1, α_2, α_3.** The framework predicts the *unified* coupling α_GUT. The breakdown into Standard Model gauge couplings at lower scales requires (a) the dim(SU(3)):dim(SU(2)):dim(U(1)) = 8:3:1 ratio (standard GUT embedding; not derived here), and (b) MSSM RG running (standard physics). Neither is part of this derivation but both are independent of A5(b).

3. **Threshold corrections.** The 1.3% discrepancy between 1/24 (tree-level) and 1/24.3 (RG-corrected) suggests the framework's natural scale is slightly below M_GUT, with sub-GUT physics contributing the remaining correction. Quantifying this would constrain the SUSY spectrum or the precise lattice scale.

4. **Cross-validation with the |Aut(K₄)| reading.** The K₄ automorphism group |Aut(K₄)| = |S₄| = 24 supplies an alternative structural angle on α_GUT = 1/24. Numerical agreement with this project's local-label-count derivation is exact; algebraic equivalence is open (see §Note above). If equivalence holds, the |S₄| reading would graduate to a second algebraic derivation; if not, |S₄| stays a numerical-only cross-check.

---


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

- `docs/theorems/theorem_car_local_jordan_wigner.md` (local CAR thm), `docs/framework/framework_axioms.md` §5b (A5(b)) — derivations
- `predictions/k_star.py` — STRICT-SOLID derivation of k* = 3
- `proofs/foundations/exponent_ladder.py` — existing project context for α_GUT vs α_1 mechanisms
- Kraft, L. G. (1949). A device for quantizing, grouping, and coding amplitude-modulated pulses. MIT thesis.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. Phys. Rev. 106, 620.
- Amaldi, U., de Boer, W., Fürstenau, H. (1991). Comparison of grand unified theories with electroweak and strong coupling constants measured at LEP. Phys. Lett. B 260, 447. (Original MSSM unification fit)
- Langacker, P., Polonsky, N. (1995). The strong coupling, unification, and recent data. Phys. Rev. D 52, 3081.
- Particle Data Group (2024) for measured M_Z gauge couplings.
