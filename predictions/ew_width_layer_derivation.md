# δ_EW^width — Derivation (the LOOP program's R-V output; registered 2026-07-02)

## 1. Abstract

We derive the electroweak radiative layer that dresses the framework's frozen α-form golden-rule
width assemblies: δ_Z = −0.4864% (= −1.81 loop units α₂/4π) on Γ_Z/M_Z and δ_W = −0.0787% on
Γ_W/m_W. The non-trivial content is not the arithmetic but the FORCING CHAIN: the loop class was
selected by pre-registered probe, the evaluation rule (which member of the KMS loop family is
physical) was DERIVED rather than assumed, the machinery was calibrated on an exactly-known EW
loop to symbolic exactness, and the value was computed blind against a pre-registered demand with
a pre-registered tier rule and falsification surfaces — and landed (pull −0.54). The layer closes
Γ_Z/M_Z from +4.8σ to −0.55σ, numerically equal to the SM's own residual. Grade is capped at
bridge-conditional by construction: the layer's numerical content is continuum-loop
(π-transcendental over K).

## 2. Framework results invoked

- **C0 (measure):** the matter sector's fluctuation measure is the quasi-free CAR-KMS(β=1) state
  (independently forced; `DN_C0_run_measure_2026-07-02.py`).
- **C2 (class):** the CAR-KMS matter loop on the P3 vertex forms is R-V's class; conditional on
  the P3/PS identification its content is standard EW on the derived site table
  (`DN_C2_vertex_loop_class_2026-07-02.py`, pre-reg 2188fbe; with T-ID1/T-ID2's derived content:
  Cl(3,1), γ⁵, the doublet table, hypercharge arithmetic).
- **V1 (evaluation rule + machinery):** `LOOP_V1_car_kms_calibration_2026-07-02.py` (pre-reg
  a5287f4). [T] the KMS occupation function is β-independent as a function only at β ∈ {0, ∞} ⟹
  exactly TWO parameter-free evaluations of the loop: the dead branch (β→0, all Pauli weights
  vanish) and the μ=0 Dirac-sea VACUUM loop (β→∞). Interior β would be a continuous free
  parameter (forbidden, CLEANROOM §7) with no invariant content (III₁, §6); a DERIVED interior
  clock is excluded at EW poles (Q1's two-sided winding no-go). The ARROW — the already-counted
  one-bit datum (T10/T16) — selects the vacuum branch. Thermality enters as statistics only
  (C1's Matsubara parity doubling). Machinery calibration: the Veltman doublet Δρ reproduced
  SYMBOLICALLY exactly; Ward/custodial/decoupling and Q_u/s²/μ²-independence exact; the massless
  optical-theorem lock Im Π_T = s(v²+a²)/(12π) at 1e-14.
- **Scheme:** `docs/framework/framework_scheme_convention.md` §7 — the framework's RG-endpoint
  couplings (g₂, s², α_s at M_Z) "live in MS̄-at-scale by definition." No scheme choice is made
  here; the convention pre-dates the loop program by ten weeks.

## 3. Derivation

**Step 1 (the object).** By V1's theorem the physical loop is the retarded vacuum EW one-loop on
the P3 vertex forms with the derived content — i.e., the standard EW radiative layer computed
with framework inputs (C2's reduction). Its effect on the α-form assembly is a multiplicative
factor (1 + δ) per width.

**Step 2 (the extraction; V2, pre-reg d37a679).** The layer is a dimensionless function of
(ŝ², α̂, α_s, m_t, M_H) with tiny derivatives at the physical point. Extract it at the one point
where a certified all-known-orders evaluation exists — the PDG 2024 EW review's SM column
(the named Type-3 worked example; archived `docs/references/pdg2024_rev_standard_model.pdf`):

$$\delta_Z^{SM} = \frac{\Gamma_Z^{SM}/M_Z^{fit}}{\text{tree}\times\text{QCD}\,(\hat s^2_Z,\;
\hat g_2^2 = 4\pi\hat\alpha/\hat s^2_Z,\;\alpha_s^{fit})} - 1 = \frac{0.0273500}{0.0274839}-1
= -0.4874\%.$$

The tree here is THIS repo's frozen S3 assembly (the exact pure functions; the leaf carries a
replica welded to them at 1e-14 by import-time asserts). Certification: Table 10.6 reassembles
Γ_had and Γ_Z at 0.03 MeV; per-channel layers (ν +0.06%, ℓ −0.115%, u +0.35%, d −0.63%,
b −2.45%) decompose into their named SM pieces; the b−d differential reproduces the Eq.-10.55
ρ_t structure with residual −0.41% = the b-mass phase space; the α-form W channel reproduces
Γ(W→eν) = 226.29 ± 0.04 MeV at +0.010%.

**Step 3 (application at framework inputs).** Every input-difference sensitivity between the
PDG point and the framework leaves is computed or bounded: the b-vertex m_t² drift (applied:
ΔS = +9.7×10⁻⁶ via Eq. 10.23's quoted scaling), and bounds for s²-curvature, the α_s tail, α̂,
and the M_H log — total |ΔS| < 0.012 loop units, thirty times below the demand band. Hence

$$\delta_Z = \delta_Z^{SM} + \Delta S = -0.4864\%,\qquad
\delta_W = \frac{\Gamma_W^{SM}/m_W^{SM}}{g^2\cdot 9/(48\pi)\times QCD_W} - 1 = -0.0787\%.$$

**Step 4 (the blind comparison; single marked block, V2).** Demand (pre-registered from S5/S6/C2,
frozen before the loop program began): −0.437% ± 0.092% = −1.62 ± 0.34 loop units. Computed:
−1.81 loop units. **Pull −0.54 — LANDING** under the pre-registered tier rule. Surfaces: Γ_W/Γ_Z
−0.06σ → +0.14σ (sub-σ, holds); pole positions untouched (rates only — the R3 clause); Γ_e = 0
exactly.

## 4. Result

$$\boxed{\;\delta_Z = -0.4864\%\;(=-1.81\ \text{loop units}),\qquad \delta_W = -0.0787\%\;}$$

## 5. Comparison with experiment

Not directly observed (registered obs = None, like δ_r). Its effect: Γ_Z/M_Z +4.76σ → −0.55σ
(Clause 8c PASS; equal to the SM's own −0.53σ); Γ_W/Γ_Z −0.06σ → +0.14σ.

## 6. Open questions

1. **The native derivation** (the grade ceiling): the layer's O(1) coefficient from the
   interacting sector coupling — the walk↔Fock dictionary / P3-PS current identification at
   theorem grade (`incomplete_equations_todo.md` §7). Until then this row is
   STRUCTURAL-DERIVATION-CONDITIONAL / SM-REPRODUCTION-CONDITIONAL (Clause 9b), never
   theorem-grade.
2. **The [external] surface:** the certified worked example's numbers (quoted verbatim in the
   leaf) are PDG-2024 fit outputs; a future PDG update shifts the layer by parts of its ±0.35‰
   internal uncertainty — the regression guards in the leaf and consumers will catch any silent
   drift and force a reviewed re-freeze.
3. **Two disclosed pre-registration size-estimate misses** (recorded in the V2 probe and the
   loop-kickoff banner; criteria held): the blanket ±2% per-channel certification gate fires on
   the b-row (−2.45% = exactly its named content); the W/Z differential is +0.41%, not ≲0.1%.
