# Master theorem doc — Substrate-Feshbach-analog dark corrections

**Date:** 2026-05-15
**Status:** MASTER reference for the framework's dark-correction mechanism.  Consolidates `theorem_dark_correction_mdl.md`, `theorem_m_nu_dark_correction_uniqueness_closure.md`, `theorem_dark_5_12_spectral.md`, and `substrate_feshbach_analog_cluster_2026-05-14.md` into a single coherent treatment.  Provides application instructions for parameter derivations needing dark corrections.
**Scope:** structural mechanism, theory, equations, application protocol.  Does NOT supersede the underlying theorem docs; supplements them with a unified entry point.

---

## 1. The mechanism — what dark corrections ARE

The framework's substrate is the srs graph (Layer 2 in `framework_architecture.md`), but the framework's underlying object is the MULTIWAY substrate (Layer 1).  The visible-sector observer is the MDL-compressed projection of the multiway substrate; everything that doesn't satisfy the MDL "pays for itself" criterion is the DARK SECTOR (Layer 6).

**Visible and dark sectors continuously exchange.**  The dark sector contains:
- The waterline-failing substrate copies (Q-space: srs-z, srs-c4, hcb-c4 — three space-group variants that are 3-regular but lose to srs on MDL bit-count).
- Uncompressed multiway branches (raw rewrites that haven't yet been compressed into a Bloch-decomposable observable).

The exchange is via the Feshbach-style self-energy of the Hashimoto walker:

$$\Sigma_Q(h) = \alpha_1^{\rm bare} \cdot \frac{\bar h}{|h|^2}$$

where $h = (\sqrt{3} + i\sqrt{5})/2$ is the walker eigenvalue at the BZ corner $P$, and $\alpha_1^{\rm bare} = (2/3)^{g-2} = 256/6561$ is the NB walker survival probability over a girth-cycle interior (theorem-grade Class A, `predictions/alpha_1.py`).

**The mass-as-flux reading**: for any Bloch eigenchannel $(k, \alpha)$, the particle's mass IS the steady-state bidirectional rewrite-flux rate between visible and dark sectors at that channel:

$$m_{(k,\alpha)} \;\propto\; \overline{\Phi^{\rm bi}_{(k,\alpha)}}$$

Massless modes (photon) have zero flux — MDL-stable on visible sector alone.  Massive modes have nonzero flux — they require ongoing dark↔visible exchange to maintain their compressed pattern.

**Dark corrections are the manifestation of this exchange in observable predictions.**  They modify the bare framework-derived couplings by a parity-odd contribution from the substrate's chirality (which enters through $h$'s complex argument).

---

## 2. The universal template

For any tree-level coupling $g$ (mass-class, gauge-class, mixing-angle-class, etc.), the framework's bridge convention asserts:

$$\boxed{\;g_{\rm physical} \;=\; g_{\rm bare} \times \biggl(1 \;-\; c_g \cdot \frac{\alpha_1^{\rm bare}}{1 - \alpha_1^{\rm bare}}\biggr)\;}$$

with two universal pieces:

### (i) The A2-T waterline winding sum

$$\frac{\alpha_1^{\rm bare}}{1 - \alpha_1^{\rm bare}} = \frac{256}{6305} \approx 0.040603$$

This is the geometric series $\alpha_1 + \alpha_1^2 + \alpha_1^3 + \cdots$ over all retained windings on the Hashimoto NB graph.  **A2-T retains every winding that pays its bits** (per `theorem_A2_mdl_from_finite_register.md`), so the sum is geometric, not single-winding.

### (ii) The coupling-specific dimensional fraction $c_g$

$c_g$ depends on the OBSERVABLE CLASS of $g$.  Each observable class corresponds to a specific substrate mechanism (Berry phase, Feshbach self-energy, cycle counting, etc.), and within that mechanism the framework derives $c_g$ from substrate primitives.

The MECHANISM determines the FORM (which parity-odd functional of $h$ enters); the SPECIFIC OBSERVABLE determines the COEFFICIENT $c_g$.

---

## 3. The parity-odd functionals of $h$

Three families appear, each tied to an observable class:

### (A) Berry-phase amplitude rotation — for dim-1 (angle/phase) observables

$$F^*_{\rm Berry}(h) = \sin(\arg h) = \frac{\mathrm{Im}(h)}{|h|} = \sqrt{5/8} \approx 0.7906$$

Used for: photon birefringence β (cosmic microwave polarization rotation).
Theorem-grade per `theorem_dark_correction_mdl.md` Lemma 2 (photon polarization is a unit vector → couples to unit-walker-phasor → parity-odd projection of $h/|h|$).

### (B) Feshbach self-energy contour — for dim-2 (mass²) observables

$$F^*_{\rm Feshbach}(h) = \mathrm{Im}\!\left(\frac{1}{h}\right) = -\frac{\mathrm{Im}(h)}{|h|^2} = -\frac{\sqrt 5}{4}$$

(The minus sign is convention-dependent; magnitude is $\sqrt{5}/4 \approx 0.5590$.)

Theorem-grade form per `theorem_m_nu_dark_correction_uniqueness_closure.md` (Feshbach contour integral over uniform Q-space density on Ramanujan circle gives residue $1/h$; parity-odd part forced by mechanism — NOT chosen by MDL).

Used for: neutrino masses $m_\nu$ — but see clarification below on the mass-as-spectral-gap reformulation. (Charged-lepton Koide chain $m_e/m_\mu/m_\tau$ is sub-leading at the Yukawa-chain level; the leading $m_e/m_\mu/m_\tau$ closure is via Family D per-leg multiway dark-disruption on the (1H+2F) Yukawa vertex, §3 (D), NOT this Feshbach class.)

**Application clarification (2026-05-15, post m_ν₃ reframing):** For neutrino masses under the 2026-05-04 substrate-spectral-gap reformulation (`m_nu3_global_spectral_gap_2026-05-04.md`), the bare formula `m_ν₃ = (k* · N_atoms) · M_Pl · N_hub^(-1/2)` ALREADY EVALUATES THE FESHBACH RESIDUE at the walker eigenvalue $h$ inside the unit disk — it IS the spectral-gap reading of $\Sigma(h) = \alpha_1 \bar h/|h|^2$ folded into the lowest-mode mass. The Feshbach mechanism is therefore **baked into the bare scale**, and the universal template's separate multiplicative factor $(1 - \sqrt{5}/4 \cdot \alpha_1/(1-\alpha_1))$ MUST NOT be applied on top — doing so would double-count the mechanism. Empirical verification: applying the multiplicative DC on top moves m_ν₃ from +0.87% (current) to −1.4% (over-correction at either sign convention) — neither closes the gap, confirming the mechanism is single-applied. The residual +0.87% on m_ν₃ (and +2.4% on m_ν₂) is the N_hub-anchor sensitivity (varies between G_F-anchored ~8.395×10⁶⁰ and m_τ-anchored ~8.44×10⁶⁰), NOT a missing Feshbach correction. The Family D sub-leading at the (0H+2F) Majorana vertex is $+\alpha_1^2/6 \approx +0.025\%$ — also negligible relative to the anchor sensitivity. See `predictions/m_nu3.py` Step 1 comment and Row P31 of `docs/parameters/parameter_uniqueness_ledger.md`.

The OLD (pre-2026-05-04) chain — `m_ν₃ = m_ν_bare^{PS} × (1 + \sqrt{5}/4 \cdot \alpha_1)` with ADOPTED-PS bare scale 0.048277 eV — applied the Feshbach factor SEPARATELY because the bare scale came from a different mechanism (PS seesaw at M_R = (2/3)^g · M_GUT). Under the new substrate-spectral-gap chain, that separation collapses. The retracted files `predictions/retracted/m_nu3_seesaw_PS.{py,md}` preserve the old chain for record.

### (C) Counting form — for dim-0 (probability/dimensionless) observables

$$F^*_{\rm counting} = 1 \quad (\text{constant, no $h$-functional})$$

Used for: V_us = 9/40 (Cabibbo angle squared), v_Higgs c = 5/12, candidate α_GUT c = 1/k_*.
The counting form's "parity-odd content" is absorbed in the integer-fraction $c$ itself; no separate $F^*$ needed.  Closure is via Sunada / Hashimoto-spectral counting (theorem-grade for v_Higgs at 5/12; hypothesis for others).

**Selection rule:** observable's TENSOR CHARACTER selects the family, then the structural counting determines $c_g$.

### (D) Per-leg multiway dark-disruption — for vertex couplings with leading-order dark correction absent (LAYER-1 HYPOTHESIS, added 2026-05-15)

A separate dark-correction family applies to vertices where the leading-order ($\alpha_1$) Class-1/2/3 dark correction is ABSENT (e.g., y_τ Yukawa vertex, λ_Higgs $|\phi|^4$ vertex — both currently listed as open in §5 / §9). The mechanism (per user 2026-05-15): dark toggles from the non-srs compressible substrate disrupt the persistence of features on srs in the multiway system. With srs-z now the dominant co-retained non-srs alternative (R-9 closure, `r9_srs_z_polynomial_derivation.py`, commit `843cfc9`, 2026-05-02, sharing g = 10 with srs), the joint NB walker survival on (srs × srs-z) gives a per-leg disruption rate at order $\alpha_1^2$ — sub-leading to the order-$\alpha_1$ template of (A)–(C).

The form (LAYER-1 HYPOTHESIS, structural reading sentinel-passing per `proofs/foundations/dark_disruption_per_leg_2026-05-15.py`):

$$c_H = \alpha_{1,\rm bare}^2 \quad \text{(per Higgs leg at vertex; joint srs × srs-z walker survival)}$$

$$c_F = -\frac{\alpha_{1,\rm bare}^2}{N_{\rm atoms} \cdot k_*} = -\frac{\alpha_{1,\rm bare}^2}{12} \quad \text{(per fermion leg; directed-edges-per-cell normalization, JW sign flip)}$$

The correction at a vertex with $n_H$ Higgs legs and $n_F$ fermion legs is:

$$\boxed{\;\frac{\delta g}{g} = -(n_H \cdot c_H + n_F \cdot c_F) \;}$$

Used for:
- y_τ vertex (1H + 2F): $\delta y_\tau/y_\tau = -(c_H + 2 c_F) = -(5/6) \alpha_1^2$
- λ_Higgs $|\phi|^4$ vertex (4H + 0F): $\delta\lambda/\lambda = -4 c_H = -4 \alpha_1^2$

Empirical match (NO fitting; all parameters framework theorem-grade):
| Observable | Empirical | Predicted (Family D) | Rel. err |
|---|---|---|---|
| $\delta y_\tau/y_\tau$ | −0.1257% | −0.1269% | +0.92% |
| $\delta\lambda/\lambda$ | −0.6007% | −0.6090% | +1.38% |
| $\lambda_{\rm obs}/y_{\tau,{\rm obs}}$ | 17.9144 | 17.9131 | **−0.007%** |

m_τ closes from +18.67σ_PDG → −0.17σ_PDG; m_H closes from +3.43σ_PDG → −0.05σ_PDG.

**Status (CORRECTED 2026-05-18, W1): Family D is THEOREM-GRADE-STRUCTURAL, conditional.** The 2026-05-15 "graduated to THEOREM-GRADE — all four routes closed" claim was wrong for $c_F$: "Routes F-1 + F-2" are *not* two independent routes (they are `canonical_encoding`-equivalent — identical via the Euler identity $2|E|=N\cdot k_*$), so the §8-rule-1 two-routes discipline is NOT satisfied for $c_F$, and stating it as such was a `parameter_linter.md` Clause-6c smuggle (unnamed MDL-bit-cost minimum conflating `canonical_encoding` with `channel_select`). $c_H$ stands (genuinely two routes: H spectral, C combinatorial). $c_F$'s correct derivation is the explicit Clause-6 two-step below, verified through the real `simulator/gating/mdl.channel_select` gate (`proofs/foundations/c_F_channel_select_waterfilling_2026-05-18.py`, commit `6c43c54`); numeric values UNCHANGED.

For $c_H = \alpha_{1,\rm bare}^2$:
- **Route H** (joint Hashimoto-spectral, `family_D_route_H_2026-05-15.py`): $c_H = q_{NB}^{2(g-2)} = (q_{NB}(\text{srs}) \cdot q_{NB}(\text{srs-z}))^{g-2}$ from joint NB walker survival.
- **Route C** (m=2 closed-bubble combinatorial, `family_D_route_C_2026-05-15.py`): $c_H = q_{NB}^{L_{\text{closed}}(m=2)}$ with $L_{\text{closed}}(m=2) = 2g - 4 = 16$ per `hashimoto_16cycle_decomposition.py`.
- Both give $q_{NB}^{16} = \alpha_{1,\rm bare}^2$ because $2(g-2) = L_{\text{closed}}(m=2)$ when seam length = 2 (srs-specific structural identity).

For $c_F = -\alpha_{1,\rm bare}^2/(N_{\rm atoms} \cdot k_*) = -\alpha_1^2/12$ — the explicit `parameter_linter.md` Clause-6 two-step L-expression (replaces the retracted "Routes F-1 + F-2 two independent routes"):

- **Step 1 — `channel_select(S, c="single_edge_spectral")`.** The channel is fixed by a structural argument BEFORE candidate enumeration: a Yukawa fermion leg is a single CAR directed-edge mode (`theorem_car_local_jordan_wigner.md §1`), structurally distinct from the gauge-singlet's democratic $2|E|$-edge sum (the $\delta_r$ channel). This selects the single-edge-spectral K-candidates and **excludes** (other channels, physically realized but coupling to other observables): the gauge-singlet object $\langle e|P_P|e\rangle/(2|E|)=1/(2|E|)^2$ (the prereg-#1 "$1/144$" — a channel mismatch, not $c_F$), and the vertex-local $1/k_*^2$ (the tree-level Yukawa-norm channel).
- **Step 2 — `canonical_encoding(S')`.** The two single-edge-channel candidates, $1/(2|E|)$ (single-edge Perron weight $\langle e|P_P|e\rangle$) and $1/(N_{\rm atoms}\cdot k_*)$ (cell directed-edge count), are **encoding-equivalent** — identical value via the Euler identity $2|E| = N_{\rm atoms}\cdot k_*$. They are NOT two independent routes; canonical (min-bit) representative: $1/(N_{\rm atoms}\cdot k_*)$.
- Result: $|c_F| = \alpha_1^2\cdot 1/(N_{\rm atoms}\cdot k_*)$, with the closed-fermion-loop sign $-1$ (Peskin–Schroeder §4.8 / `theorem_car_local_jordan_wigner.md`). $\delta_r$-anchor consistency: the SAME gate on the gauge-singlet channel returns $1/(2|E|)$ (the proven $\delta_r$ value).

The clean-rational test (§8 rule 4) is satisfied: $c_H, c_F \in K = \mathbb{Q}(\sqrt 2, \sqrt 3, \sqrt 5)$ (here $c_F\in\mathbb{Q}\subset K$). **§8 rule 1 (two independent routes) is satisfied for $c_H$ only** (Route H spectral + Route C combinatorial, genuinely distinct). For $c_F$ it is **NOT** satisfied — the historical "F-1/F-2" are `canonical_encoding`-equivalent, one derivation expressed two ways. $c_F$ closes at **THEOREM-GRADE-STRUCTURAL, conditional** on the Step-1 channel argument (a structural argument at $\delta_r$'s tier, `theorem_unified_oblique.md §6.1` — not a from-resolvent theorem). Family D as a whole is therefore THEOREM-GRADE-STRUCTURAL (conditional), not UNIQUE-THEOREM-GRADE.

v_Higgs calibration check (§8 rule 2): Family D at v_Higgs (1 Higgs leg) predicts $\delta v/v = -c_H = -\alpha_1^2 \approx -0.152\%$ as a sub-leading correction to the leading $-5/12 \times \alpha_1/(1-\alpha_1)$ (Family C). This sub-leading correction is **absorbed into the N_hub anchor calibration** via the G_F round-trip (`predictions/v_higgs.py`, `predictions/N_hub.py`) — consistent by construction with the framework's existing v-sector closure.

Per §8 rule 6, the (now THEOREM-GRADE-STRUCTURAL-conditional) closures are propagated to children predictions, numeric values unchanged: m_τ → −0.17σ_PDG (from +18.67σ_PDG), m_H → −0.05σ_PDG (from +3.43σ_PDG; m_H rides $c_H$ only — not affected by the $c_F$ conditional), m_e and m_μ inherit m_τ's absolute scale via the (theorem-grade) Koide ratios.

**Family D non-applicability to quark sector (2026-05-15 audit).** Quark Yukawa vertices are structurally 1H + 2F (same vertex topology as y_τ) and would receive δy_q/y_q = −(5/6)·α₁_bare² ≈ −0.127% if a tree-level y_q existed. But:
- Row P37 (koide_quark_ratio = 14/5) is purely g-dependent with no α₁ dependence; Family D doesn't enter, and its −0.6% residual is a structural artifact of the n=1/n=2 Cl(6) Fock expansion, not a missing dark correction.
- Row P38 (m_top) was RETRACTED 2026-05-04 — used PDG m_c, m_b as empirical inputs, failing zero-empirical-inputs.
- Row P39 (m_u/d/s/c/b) is BLOCKED on quark Yukawa structure; the upstream blocker is Need-D-3 (Y_u vs Y_d distinct eigenbasis on C³_gen) with 9 ruled-out attacks and a two-layer block requiring framework extension beyond M ⋊_α Z_3. Until Need-D-3 closes, there is no tree-level quark Yukawa for Family D to correct multiplicatively. The koide_quark_ratio's residual cancels under Family D (Family D applies identically to up and down vertices), so even a future Need-D-3 closure won't have Family D bridging Row P37's residual.

The calibration discipline (§8 rule 2, "must reproduce $c_v = 5/12$ for v_Higgs") does NOT apply directly because Family D is at a DIFFERENT ORDER ($\alpha_1^2$) than the universal template ($\alpha_1$); v_Higgs has its leading-order Family (C) closure at $c_v = 5/12$ which is unrelated to per-leg dark disruption. However, Family D should be CONSISTENT with v_Higgs in the sense that the per-leg sub-leading correction on v_Higgs from this mechanism is negligible (1 Higgs leg via VEV → δv/v = −α₁² ≈ −0.15%, currently absorbed in the N_hub anchor calibration and not separately tested).

Family D status (CORRECTED 2026-05-18, W1):
1. **Route H (Hashimoto-spectral) for c_H:** CLOSED. Joint NB walker on (srs × srs-z) gives $c_H = q_{NB}^{2(g-2)} = \alpha_1^2$. See `family_D_route_H_2026-05-15.py`.
2. **Route C (combinatorial-counting) for c_H:** CLOSED. m=2 closed-bubble length $L_{\text{closed}}(m=2) = 2g-4 = 16$ on srs gives $c_H = q_{NB}^{L_{\text{closed}}} = \alpha_1^2$. See `family_D_route_C_2026-05-15.py`. *(Routes H, C are genuinely independent → §8-rule-1 satisfied for $c_H$.)*
3. **$c_F$ via the Clause-6 two-step** (above): `channel_select` (single_edge_spectral channel, fixed by `theorem_car_local_jordan_wigner §1`) → `canonical_encoding` ({$1/(2|E|)$, $1/(N\!\cdot\!k_*)$} encoding-equivalent via Euler $2|E|=N\!\cdot\!k_*$) → $c_F=-\alpha_1^2/(N\!\cdot\!k_*)$. Verified via the real `simulator/gating/mdl.channel_select` gate in `proofs/foundations/c_F_channel_select_waterfilling_2026-05-18.py`. The historical `family_D_route_{F,F2}_2026-05-15.py` are the two `canonical_encoding`-equivalent expressions — **SUPERSEDED** by the explicit two-step (kept as record; they were the Clause-6c smuggle).

§8-rule-1 (two *independent* routes) is satisfied for $c_H$ only. $c_F$ is THEOREM-GRADE-STRUCTURAL, conditional on the Step-1 channel argument ($\delta_r$'s tier, not from-resolvent). Family D is THEOREM-GRADE-STRUCTURAL (conditional), **not** UNIQUE-THEOREM-GRADE. Numeric outputs unchanged ($\delta y_\tau/y_\tau=-(5/6)\alpha_1^2$, $\delta\lambda/\lambda=-4\alpha_1^2$).

Numerical predictions in children files (m_τ, m_e, m_μ, m_H, y_τ, λ_Higgs): per §8 rule 6, theorem-grade c_g values ARE propagated to children. Specifically:
- y_τ_corrected = y_τ_tree × (1 - (5/6) α₁²) ≈ 7.21647×10⁻³ vs obs 7.21655×10⁻³  (−0.01% residual = -0.01σ_PDG class)
- λ_corrected = λ_tree × (1 - 4 α₁²) ≈ 0.129269 vs obs 0.129281  (−0.01% residual = -0.04σ_PDG class)
- m_τ_corrected = m_τ_tree × (1 - (5/6) α₁²) ≈ 1.7768 GeV vs obs 1.77686 GeV  (−0.17σ_PDG)
- m_H_corrected = m_H_tree × √(1 - 4 α₁²) ≈ 125.195 GeV vs obs 125.20 GeV  (−0.05σ_PDG)
- m_e, m_μ inherit m_τ correction via Koide ratios.

---

## 4. The two derivation routes for $c_g$ (calibrated case: $c_v = 5/12$)

For v_Higgs, $c_v = 5/12$ is **theorem-grade** via TWO INDEPENDENT derivations that reach the same number.  These are the calibrating routes for any new dark-correction derivation.

### Route H — Hashimoto-spectral fraction

$$c = \frac{\dim(\text{marginal Hashimoto sector at coupling vertex})}{\dim(\text{full NB sector at vertex})}$$

For v_Higgs: $c = (2(|E|-|V|) + 1)/(2|E|)$ on srs ($|V|=4, |E|=6$) gives $5/12$.

The "marginal sector" is the $|\lambda| = 1$ subspace of the Hashimoto operator $B$ — eigenvalues neither Perron ($|\lambda| > 1$) nor oscillatory-decaying.  Per Stark-Terras spectral theory, marginal-mode dimension on a $k$-regular graph at $\Gamma$ is $2(|E|-|V|) + 1$.

Doc: `theorem_dark_5_12_spectral.md`.

### Route C — Cycle-counting fraction

$$c = \frac{n_X}{N_{\rm atoms} \cdot k_*^2}$$

For v_Higgs: $c = n_g / (N_{\rm atoms} \cdot k_*^2) = 15 / 36 = 5/12$.

The numerator $n_X$ counts specific substrate cycles tied to the coupling vertex (for v_Higgs: $n_g = 15$ unoriented girth cycles per vertex, Sunada 2012 + DFS).  The denominator $N_{\rm atoms} \cdot k_*^2$ is the per-vertex coupling-pair count from A2 edge process (Higgs F0: A2 is an edge process → couples through ALL $k_*^2$ ordered edge pairs at each vertex).

Doc: `theorem_dark_correction_mdl.md` + `proofs/foundations/dark_feshbach_a2_closure.py`.

### Calibrating constraint

**Both routes give the same number ($5/12$).**  Any structural derivation of $c$ for a NEW observable must reproduce $5/12$ for v_Higgs via the same mechanism.  This is the calibration discipline against numerology: deriving a number that fits the data of the new observable but doesn't reproduce v_Higgs's $5/12$ does NOT count as evidence.

---

## 4.5 Meta-classification — vertex-level vs propagator-level corrections (added 2026-05-15 EOD+1)

Families A–D above (Berry, Feshbach, counting, per-leg multiway) all
share a key STRUCTURAL feature: the correction is **sign-uniform** —
the multiplicative factor $g_{\rm physical}/g_{\rm bare}$ takes a
definite sign (typically suppression) determined by the mechanism, and
applies to ALL observables sharing the same substrate structure.

**Family D's success on Yukawa/Higgs vertices** confirmed this for
*vertex-level* observables (single interaction point with definite
leg-count topology).  The Yukawa vertex (1H + 2F) and the $|\phi|^4$
vertex (4H + 0F) take corrections $-(5/6)\alpha_1^2$ and $-4\alpha_1^2$
respectively — both negative, both sign-uniform.

**Family D's failure on the gauge-boson 2-point function** (M_Z, m_W
single-session probe 2026-05-15 EOD+1, `M_Z_m_W_family_D_probe_2026-05-15.py`)
reveals the limit of this template:

  - The empirical residual on M_Z² is $-0.71\%$ (M_Z too high).
  - The empirical residual on m_W² is $+0.33\%$ (m_W too low).

These have OPPOSITE SIGNS.  Sign-uniform Family-D corrections cannot
produce this split.  Through the tree relation $m_W = M_Z \cos\theta_W$
any multiplicative correction propagates identically to both.

### Meta-classification

| Class | Observable types | Sign convention | Examples (closed) | Examples (open) |
|---|---|---|---|---|
| **Vertex-level** | single-vertex couplings: Yukawa, $|\phi|^4$, gauge coupling at unification | sign-uniform; per-leg counting; Families A–D | y_τ, λ_Higgs (Family D); α_GUT (Family C); v_Higgs (Family C) | — |
| **Propagator-level (custodial-symmetric)** | gauge boson 2-point under custodial SU(2)_L × SU(2)_R | sign-uniform; inherits family from "what fills the loop" | (none yet) | M_Z scale residual S |
| **Propagator-level (custodial-breaking)** | observables sensitive to top-bottom Yukawa asymmetry | requires asymmetric mechanism; Families A–D do NOT apply | (none in framework) | M_Z/m_W ρ-parameter shift R ≈ +1.05% |
| **Mass-matrix block** | mixing angles, mass ratios with internal block structure | $\tan^2(\arg h)$, $\cos(\arg h)$ style | θ_23, θ_13 | — |
| **Sub-sector projection** | observables with distinct sub-sector splits | not single $F^*$ | Λ_CC Path B | — |

### Selection rule extension to §6 Step 1

When the observable is propagator-level (gauge boson 2-point, fermion
self-energy, ...), the application protocol must decompose the
residual into:

  $\delta(g_{\rm obs})/g = S_{\rm scale} + R_{\rm asymmetric}$

where:
- $S_{\rm scale}$ is the sign-uniform piece — candidate for Families A–D
- $R_{\rm asymmetric}$ is the custodial / sector-asymmetry piece —
  requires a *Family E* mechanism not yet identified

If $R_{\rm asymmetric}$ is non-zero (test: do related observables
sharing the same propagator have residuals of OPPOSITE SIGN?), the
substrate mechanism MUST include access to the relevant asymmetric
sector (e.g., top-bottom Yukawa for Δρ; B−L charge for some β-class).
Families A–D alone are insufficient.

### Empirical decomposition for M_Z / m_W (2026-05-15 EOD+1)

Decomposing the M_Z + m_W joint residual under the ρ-parameter invariant
($\rho \equiv m_W^2 / (M_Z^2 \cos^2\theta_W)$, tree value = 1):

  $S_{\rm scale} = (M_{Z,\rm obs}/M_{Z,\rm pred})^2 - 1 = -0.71\%$ (Family-B-like)
  $R_{\rm rho} = \rho_{\rm obs} - 1 = +1.05\%$ (custodial-breaking)

**Closure paths:**
- $S_{\rm scale}$: Family B (Feshbach) probe `M_Z_family_B_probe_2026-05-15.py`
  reduces residual ~10× but $c_{M_Z}$ not closed at clean rational.
  Status: family assignment plausible, structural $c$ NOT graduated.
- $R_{\rm rho}$: Family E candidate — substrate analog of SM Δρ via
  top-bottom Yukawa asymmetry.  the author's separate private derivation-style h^n + GJ=3 + y_t(GUT)=1
  reaches 0.95% at GUT (A4 probe `A4_h_n_delta_rho_probe_2026-05-15.py`),
  within 10% of empirical 1.05%.  **Not closed** — requires y_t(GUT)=1
  graduation + substrate loop normalization.

### Family E — custodial-symmetry-breaking corrections (RESOLVED 2026-05-15 EOD+16, THEOREM-GRADE-CONDITIONAL on c=1/2)

**Status: identified as a single Hashimoto SPECTRAL mechanism — NOT a
Family-D composition and NOT a c_S+c_E superposition.**  The earlier A4
reading $\Delta\rho \approx (3/(32\pi^2))(1 - 9 y_\tau^2)$ is **rejected**:
$1/(32\pi^2) \notin K = \mathbb{Q}(\sqrt2,\sqrt3,\sqrt5)$ — it violates the
O9 algebraicity meta-theorem (a continuum loop factor cannot appear in a
substrate-native quantity).  Phase C (`proofs/foundations/family_E_phase_C_
spectral_delta_rho_2026-05-15.py`) supersedes it.

**The mechanism.**  M_Z², m_W² are mass² observables → mass²-class Feshbach
functional $F = \mathrm{Im}(h_P)/|h_P|^2 = \sqrt5/4$ (the SAME functional
m_ν uses; calibration-locked).  The Hashimoto operator $B_{NB}(\mathrm{srs})$
is Ramanujan-saturated: $|h_P|^2 = k^*-1 = 2$ EXACTLY $=$ Perron magnitude.
Therefore the Z residue (Perron, real, species-conserving $n\to n$,
$n_{\rm fixed}=0$) and the W residue ($h_P$, phase, species-changing
$n{=}1\leftrightarrow n{=}2$, $n_{\rm fixed}=2$ Feshbach scattering) have
**equal modulus** — the entire custodial splitting is the *phase* of the
one eigenvalue $h_P$.  The Z piece is custodial-symmetric and **cancels in
the $\rho$ ratio** (this is how Family C "interacts" with E — by cancelling,
not adding); the W phase-piece carries $\delta\rho$:

$$\boxed{\;\delta\rho \;=\; c \cdot \frac{\mathrm{Im}(h_P)}{|h_P|^2} \cdot \alpha_{1,\rm bare} \;=\; \tfrac12 \cdot \tfrac{\sqrt5}{4} \cdot \left(\tfrac23\right)^{8} \;=\; +1.091\%\;}$$

vs observed $+1.043\%$ (**+4.6%**, zero fitting, sign correct).  $\alpha_{1,\rm
bare}$ is forced by the Feshbach Exponent Principle (W self-energy =
$n_{\rm fixed}=2$ scattering).  The coefficient $c=1/2$ has two converging
substrate-counting readings — $1/(k^*-1)$ (per-NB-forward-choice carrying
the species flip) and $2/N_{\rm atoms}$ (W$^\pm$ pair over 4 cell atoms).

**Residual open piece — CLOSED 2026-05-15 EOD+16 (Phase C.1).**  $c=1/2$
is rigorously derived: it is NOT a substrate counting coefficient but the
squared W-field normalization that appears structurally in $\rho$ itself.
With $m_V^2 \propto g_V^2 \Pi_V$,

$$\rho = \frac{g_W^2 \Pi_W}{g_Z^2 \Pi_Z \cos^2\theta_W} = \frac{(g/\sqrt2)^2\,\Pi_W}{(g/\cos\theta_W)^2\,\Pi_Z\,\cos^2\theta_W} = \tfrac12\cdot\frac{\Pi_W}{\Pi_Z}$$

so $c = g_W^2/(g_Z^2\cos^2\theta_W) = (g/\sqrt2)^2/g^2 = 1/2$ EXACTLY,
$\theta_W$-independent — a DEFINITIONAL electroweak constant ($W^\pm =
(W^1\mp iW^2)/\sqrt2$), at the SAME Type-3 tier as the $m_W = M_Z\cos\theta_W$
tree relation already used in `predictions/m_W.py`.  Two independent
routes converge (the Family-C two-routes standard): (R1) the EW gauge-field
normalization above; (R2) α2'''-PIVOT consistency — the SAME $1/2$ makes
$\mathrm{Tr}[T_+T_-]/\mathrm{Tr}[T_3^2]=4/2=2$ give $\rho_{\rm tree} =
(1/2)\cdot2 = 1$ exactly (the known custodial-preserved tree result).  The
earlier $1/(k^*-1)$ and $2/N_{\rm atoms}$ readings are DEMOTED as
coincidence (no structural tie to $\rho$'s coupling normalization).
Probe `proofs/foundations/family_E_phase_C1_c_half_W_normalization_2026-05-15.py`
(all 4 pre-declared checks PASS).

**Grade:** the custodial-breaking δρ mechanism is **THEOREM-GRADE-
STRUCTURAL** — every factor rigorously originated (c=1/2 Type-3 EW
definitional; √5/4 m_ν-calibrated Feshbach; α₁_bare Feshbach Exponent).
Clause 7 (derivation rigor) PASSES; Clause 8 on δρ matched to +4.58%
relative (plausibly subleading spectral corrections beyond the leading
$h_P$ residue).  No remaining c-conditional.  The SEPARATE absolute-M_Z
residual is upstream M_unif (cancels in the scale-independent δρ).

**Why this resolves the §4 open question:** Family E is a NEW mechanism
(distinct from A–D) but it is *spectral*, not a per-leg counting family —
it lives at the propagator level as a residue of $B_{NB}$ at the Ramanujan
eigenvalue, with the mass²-class Feshbach functional selecting Im$(h_P)$.
It is K-rational (respects O9), unlike the rejected $1/(32\pi^2)$ reading.
The selection rule (§"Selection rule extension"): propagator-level
custodial-breaking observable → Family E spectral residue at $h_P$ with
mass²-class Feshbach functional.

**UNIFIED-OBLIQUE THEOREM (2026-05-16) — Family E and the Family-C
oblique are TWO eigen-channels of ONE $B_{NB}$.**
`docs/theorems/theorem_unified_oblique.md` +
`proofs/foundations/unified_oblique_one_resolvent_2026-05-16.py`.
The custodial-breaking $\delta\rho$ (this section, W/$h_P$ channel) and
the sign-uniform absolute-$M_Z$ oblique $\delta_r$ (Z/Perron channel,
ledger Row P64, `predictions/delta_r.py`) are the two gauge-vertex
projections of the **same** resolvent
$G_{NB}(u)=(I-u\,B_{NB}(\mathrm{srs}))^{-1}$:

- **Z neutral** (species-conserving) → **Perron** eigenvalue
  ($\lambda_P=k^*-1=2$, *dominant*, uniform eigenvector) → Family-C
  universal template with coefficient $c_S=1/(2|E|)=1/12$;
- **W charged** (species-changing) → $h_P$ ($|h_P|^2=k^*-1$,
  *sub-dominant*, phase) → Family-E mass²-Feshbach $\tfrac12\sqrt5/4$.

The **new theorem-grade content** is the derivation of $c_S$: it is the
gauge-singlet projection of the $B_{NB}$ Perron-eigenvalue residue,
$c_S=1/(2|E|)=1/12$ (the uniform vector is the verified Perron
eigenvector $B_{NB}\mathbf 1=(k^*-1)\mathbf 1$). The two historical
readings — Route H $1/(2|E|)$ and Route C $k^*/(Nk^{*2})=1/(Nk^*)$ —
are the SAME number by the **handshake lemma** $2|E|=N k^*$ (graph
identity, not coincidence). This **replaces** the retracted Phase-A fit
(`family_E_phase_A_S_scale_gauge_2point_2026-05-15.py`, stale base
predictions) flagged by the `parameter_linter` Checkpoint-1 triage. The
Perron-dominance-vs-$h_P$-subdominance form selection — previously only a
structural argument — is now **DERIVED** (2026-05-16, `theorem_unified_oblique.md`
§7.5): the tree cavity resolvent $g(z)=1/(z-k\,f(z))$ *is* the Dyson
resummation, and its analytic structure (neutral $z{=}k$ off the McKay
support, discriminant $>0$ ⇒ geometric series converges ⇒ Family-C
$\alpha_1/(1-\alpha_1)$; on-shell $z{=}2\sqrt q$, discriminant $=0$ ⇒ branch
point ⇒ leading-only ⇒ Family-E $\alpha_1$) derives the dichotomy. This
closes the last Clause-7 rigor gap; the same tree cavity GF also closes the
tree-cover **S** ($g(k){=}u^*{=}2/3$ off-support, $g(2\sqrt q){=}\sqrt q$;
$S=\tfrac1{12}(\sqrt2-\tfrac23)\alpha_1/(1-\alpha_1)=+0.253\%$). Remaining
gaps are purely Clause-8 numerical (named +4.58% $\delta\rho$ residual;
absolute-mass $\sigma_{\rm PDG}$ floor); the $c_S$ piece is theorem-grade.

**Selection-rule re-audit (2026-05-16; `theorem_unified_oblique.md` §7.6,
`proofs/foundations/selection_rule_reaudit_2026-05-16.py`).** With the
form-selection rule now *derived*, every propagator-level member of the §5
catalogue was re-audited via the Ihara map $\lambda=h+q/h$. **No
misassignment** — the resummed-vs-leading taxonomy of this section
(previously observable-class heuristics + the v_Higgs $c{=}5/12$
calibration anchor) is now **derived-consistent**: off the McKay support
($\lambda{=}{\pm}k$, disc $>0$: v_Higgs, α_GUT, $\delta_r$, S) ⇒ resummed
Family-C; on the McKay cut (disc $\le0$: $\delta\rho$, m_ν₃, β, θ₂₃, U) ⇒
leading Family-E/Feshbach. Numerical impact zero (no reassignment, not
manufactured); the v_Higgs anchor is independently *reproduced* by the
criterion it was never given. **Criterion sharpened:** the rule is disc
$\le0$ (the whole cut), not "the band edge" — $h_P$ sits at *interior*
$\lambda=\sqrt3$ (disc $=-5$). **Corollary** (constrains the open
$\delta\rho$ +4.58%): $\delta\rho$ is on the cut, so a resummation
$1/(1-\alpha_1)$ closure is *forbidden*; the residual must be a sub-tree
multi-insertion sum. Family-D vertex per-leg ($\propto\alpha_1^2$) is a
distinct mechanism, correctly out of scope.

**FLAVOR EXTENSION (2026-05-16) — the CKM triple is the off-diagonal
reading of the same $B_{NB}$.** `theorem_unified_oblique.md` §8 +
`proofs/foundations/quark_unification_over_determination_test_2026-05-16.py`
(6/6 pre-declared aborts; bound to live `match`/`CountingKernel` so "same
B" is provable). The δρ W/$h_P$ channel of *this* section and the CKM
amplitudes are readings of the **same** operator at the **same** datum
($a=(2/3)^8$ = the Feshbach W1 $n_{\rm fixed}{=}2$ coupling; $h_P$): δρ =
bare $a$ × Feshbach $\sqrt5/4$ × $c{=}\tfrac12$; $\delta_r$ & $V_{cb}{=}256/6305$
are *provably the same* resolvent-resummed $a/(1{-}a)$ under two
projections (Perron $1/12$ vs unit); $V_{ub}$ (multi-cycle host-sum, same
$q_{NB}$), $V_{us}{=}9/40$ (counting projection) — five independently
theorem-grade observables, zero fitted constants. The bare↔resummed link
is the $(I-\,\cdot\,)^{-1}$ geometric series, *forced not fitted*.
**Grade THEOREM-GRADE-STRUCTURAL** (structural cross-lock; no number, no
grade of Rows P3/P4/P14 changed; not theorem-grade-numerical). This
dissolves the Need-D-3 eigenbasis-misalignment framing *as a mechanism
question*; the 3×3 generation/$C_{36}$-twist labeling stays as the
data-anchored non-blocking residue (reframed as the resolvent's index
structure, not eliminated), and the up-sector $y_t$ natural-scale anchor
remains the single hard residue.

---

## 5. Catalogue — cluster of substrate-Feshbach-analog observables

(Updated 2026-05-15 to include α_GUT.)

| Observable | Bare | $c$ form | Status | Mechanism family |
|---|---|---|---|---|
| v_Higgs | $\delta^2 M_{\rm Pl}/(\sqrt 2\, N_{\rm hub}^{1/4})$ | **5/12** | **theorem-grade** | counting (Route H + Route C) |
| m_ν_3 | substrate spectral gap $= (k^* N_{\rm atoms}) M_{\rm Pl}/\sqrt{N_{\rm hub}}$ | **mechanism baked in** (no separate multiplicative factor; see §3 (B) clarification) | theorem-grade-conditional (Feshbach mechanism via spectral-gap reading) | Feshbach (uniqueness theorem) |
| β cosmic birefringence | framework $\alpha_{\rm EM}(M_Z)$ | $\sin(\arg h) = \sqrt{5/8}$, **c = 1** | **THEOREM-GRADE-STRUCTURAL** (DOWNGRADED 2026-05-16: the observed-$\alpha_{\rm EM}$ substitution was a smuggle — node now wired to framework $\alpha_{\rm EM}$, α(0) Δα Clause-9-blocked = named gap; FORM theorem-grade, NUMBER framework-α_EM-conditional; β=0.354°/+0.13σ replaces retracted 0.331°; Row P44) | Berry phase |
| λ_Higgs | $2 \alpha_1^{\rm full}$ | **Family D: $\delta\lambda/\lambda = -4\alpha_1^2$** (4H legs at \|φ\|⁴ vertex) | **THEOREM-GRADE** (2026-05-15 EOD: Routes H + C closed for c_H; Routes F-1 + F-2 closed for c_F; m_H closes to −0.05σ_PDG) | per-leg multiway dark-disruption (Family D §3) |
| y_τ | $\alpha_1^{\rm full}/k_*^2$ | **Family D: $\delta y_\tau/y_\tau = -(5/6)\alpha_1^2$** (1H+2F at Yukawa vertex) | **THEOREM-GRADE** (2026-05-15 EOD: same as λ_Higgs; m_τ closes to −0.17σ_PDG) | per-leg multiway dark-disruption (Family D §3) |
| **α_GUT** (added 2026-05-15, graduated 2026-05-15 EOD+1) | $1/(2^{k_*} k_*) = 1/24$ | **$c = 1/k_*$** | **THEOREM-GRADE** (Routes H + C closed; selection rule substrate-derived) | counting (cycle-only marginal sector) |
| θ_13 PMNS | mixing block | structurally $F^* = 1$ (after parity projection) | derived (semi-structural) | mass-matrix block |
| θ_23 PMNS | mixing block | $\tan^2(\arg h) = 5/3$ | derived | mass-matrix block |
| V_us | $k_*^2/(g \cdot N_{\rm atoms}) = 9/40$ | n/a — counting density, NOT a Feshbach analog | theorem-grade | direct A2 counting |
| Λ_CC w_eff | not single c | V_Ram h↔h̄ split (4+4) | distinct mechanism | sub-sector projection |
| **M_Z scale / $\delta_r$** (added 2026-05-15 EOD+1; **CLOSED 2026-05-16**) | $\sqrt\pi v \sqrt{\alpha_2 + (3/5)\alpha_1}$ | $c_S = 1/(2|E|) = 1/12$ | **THEOREM-GRADE-STRUCTURAL** ($c_S$ Perron-residue derivation theorem-grade; unified-oblique Z/Perron channel, `theorem_unified_oblique.md`) | Family-C universal template (Perron channel of the one $B_{NB}$) |
| **ρ-parameter / $\delta\rho$** (added 2026-05-15 EOD+1; **CLOSED 2026-05-15 EOD+16 / unified 2026-05-16**) | $1$ tree level | $\delta\rho = \tfrac12\cdot\tfrac{\sqrt5}{4}\cdot(2/3)^8$ | **THEOREM-GRADE-STRUCTURAL** (Phase C/C.1; the rejected $3/(32\pi^2)$ A4 reading is $\notin K$) | Family-E mass²-Feshbach ($h_P$ channel of the one $B_{NB}$) |
| **U (PT oblique)** (added 2026-05-16) | n/a (slope difference) | $U\approx 0$, $|U|\lesssim\alpha_1|S|$ | **THEOREM-GRADE-STRUCTURAL** (`theorem_unified_oblique.md` §7.1) — $\sqrt{k^*{-}1}$ Ramanujan sector scale-FROZEN $\Gamma\!\to\!P$; matches robust $|U|\!\ll\!|S|,|T|$ | derivative-class; same $B_{NB}$, scale-invariant sector |
| **Δκ (eff. mixing)** (added 2026-05-16) | n/a | $\Delta\kappa_{\rm lead}=\tfrac{c_W^2}{c_W^2-s_W^2}\delta\rho\approx+1.53\%$ | **inherits δρ grade** (definitional Type-3 recomb.; `theorem_unified_oblique.md` §7.2) — full $\sin^2\theta_{\rm eff}$ diff SM-scheme-confounded (named) | Type-3 EW recombination of δρ (no new object) |
| **S (PT oblique)** (added 2026-05-16; **CLOSED 2026-05-16 tree-cover**) | neutral Perron-channel $\Gamma\!\to\!P$ flow (tree cover) | $S=\tfrac1{12}(\sqrt2-\tfrac23)\tfrac{\alpha_1}{1-\alpha_1}=+0.253\%$ | **THEOREM-GRADE-STRUCTURAL** (`theorem_unified_oblique.md` §7.5): rigorous tree cavity GF — $g(k){=}u^*{=}2/3$ off-support finite (resolves cell Perron divergence), $g(2\sqrt q){=}\sqrt q$; K-rational, δ_r/δρ-class, no fit | derivative-class; same $B_{NB}$, tree-cover flow |
| **Δα (photon Π_γγ)** (added 2026-05-16) | photon channel: charge-weighted Perron/off-support $\Gamma\!\to\!P$ flow | — | **BLOCKED** (`substrate_Delta_alpha_blocked_verdict_2026-05-16.md`): Δα_had analog B1-scoping-NEGATIVE (multiway+R-14); Δα_lep has no first-principles-FORCED K-rational photon coeff (closest −3.6% = cherry-pick, Clause-9 numerology fail; SM value lepton-mass-log transcendental). `delta_alpha_running=9.092` tagged Clause-9 STRUCTURAL-DERIVATION-CONDITIONAL (value unchanged). Clean-ratio: this is the SECONDARY R_∞-residual piece; dominant = α_EM(M_Z) gauge-cluster drift | photon channel of the one $B_{NB}$ — BLOCKED (Clause 9 + B1 wall) |

**Observables with NO dark correction:** V_cb (Level-3 walker amplitude, already includes waterline sum), m_e/m_μ ratios (Koide algebra cancels by construction), A_hemispherical (no cycle amplitude), η_B (channel-select on $K = \mathbb{Q}(\sqrt 2, \sqrt 3, \sqrt 5)$ structurally complete), M_R / M_unif (substrate-local-family mass scales — parity not violated at unbroken-PS scale).

---

## 6. APPLICATION PROTOCOL — how to use dark corrections for a new observable

When deriving a parameter, follow this protocol:

### Step 1 — Identify the observable class

Determine the tensor character of the observable:

| Tensor character | Mechanism | $F^*$ family | Examples |
|---|---|---|---|
| dim-1 (angle/phase, unit-bounded) | Berry phase | $\sin(\arg h)$ | β |
| dim-2 (mass², self-energy correction) | Feshbach contour | $\mathrm{Im}(h)/|h|^2$ | m_ν |
| dim-0 (probability, dimensionless) | counting (Sunada / spectral) | $1$ (no h-functional; c absorbs structure) | v_Higgs (5/12), α_GUT candidate (1/k_*) |
| Vertex coupling (Yukawa, $|\phi|^4$) | per-leg multiway disruption (Family D) | sign-uniform; $c_H$, $c_F$ counted per leg | y_τ, λ_Higgs |
| Mass-matrix element (2×2 block) | block-diagonalization | $\tan^2(\arg h)$ etc. | θ_23, mass ratios |
| **Propagator scale (gauge boson mass)** | Family B candidate (sign-uniform shift) | TBD | M_Z scale residual (open) |
| **Propagator custodial-breaking** | **Family E (provisional; new mechanism needed)** | asymmetric; not single $F^*$ | M_Z/m_W ρ-shift, Δρ-analog (open) |
| Sub-sector projection | V_Ram split | not single $F^*$ | Λ_CC Path B |

**Watch out:** if the observable doesn't cleanly fit one of these families, the framework's dark-correction mechanism may NOT apply.  Don't force it.

### Step 1b — Tensor-character + family-fit declaration (added 2026-05-15 EOD+2)

Per §4.5 vertex-vs-propagator meta-classification: an observable's tensor
character determines whether Family A/B/C/D (sign-uniform per-leg counting)
can apply, or whether the observable needs Family E (custodial-breaking,
asymmetric).

**Vertex-level observables** (single interaction point with sign-uniform
per-leg topology): use Family A/B/C/D as appropriate to tensor character.

**Propagator-level observables** (gauge boson 2-point, fermion self-energy)
require:
- Decompose residual: $\delta g/g = S_{\rm scale} + R_{\rm asymmetric}$
- $S_{\rm scale}$ takes Family A/B/C/D
- $R_{\rm asymmetric}$ requires Family E (custodial-breaking; not yet derived)

Per `parameter_linter.md` Clause 9 (9c): a derivation that uses the WRONG
family for the observable's tensor character is a Clause 9 violation
(family-assignment gap).

### Step 1c — Algebraicity test (added 2026-05-15 EOD+2)

Per `theorem_lattice_coupling_broader_implications.md` §1a + §1b, the
closure form must sit in K = ℚ(√2, √3, √5).  If the derivation cites a
Type-3 SM mechanism whose value involves continuum loop factors
(1/(16π²), Sirlin Δr, Δα_had, etc.), perform the π-audit per Clause 9:
either derive the K-rational substrate analog or tag as
STRUCTURAL-DERIVATION-CONDITIONAL.

The bridge-attribution-as-closure pattern is a Clause 9 violation.
**Canonical exemplar:** SM 2-loop EW bridge attribution for M_Z/m_W
(commit f878f82 retracted 4ce4d5c, 2026-05-15).

### Step 2 — Decide if dark correction applies at all

NOT all observables get dark corrections.  Skip if:

- The observable is at the **unbroken-PS scale or above** (M_R, M_unif, v_BZJ bare value).  Parity not violated; no sector for $\sin(\arg h)$ or $\mathrm{Im}(h)/|h|^2$ to couple to.
- The bare derivation already includes the waterline sum (V_cb's $256/6305$ already absorbs it).
- The observable is a RATIO that cancels by construction (m_e/m_μ via Koide).
- The observable is a counting density without a walk amplitude (V_us = 9/40).
- The observable lives at the multiway level (Layer 1, not Layer 2) — requires NA-4 closure first.

### Step 3 — Apply the universal template

$$g_{\rm physical} = g_{\rm bare} \times \biggl(1 - c_g \cdot \frac{\alpha_1^{\rm bare}}{1 - \alpha_1^{\rm bare}}\biggr)$$

with $\alpha_1^{\rm bare}/(1-\alpha_1^{\rm bare}) = 256/6305$ (universal).

### Step 4 — Derive (or hypothesize) the dimensional fraction $c_g$

Use one of the derivation routes:

- **Route H** (Hashimoto-spectral): $c = \dim(\text{relevant sector}) / \dim(\text{total NB})$.  For v: $(2(|E|-|V|)+1)/(2|E|) = 5/12$.
- **Route C** (cycle-counting): $c = n_X / (N_{\rm atoms} \cdot k_*^2)$.  For v: $n_g/(N_{\rm atoms} \cdot k_*^2) = 15/36 = 5/12$.

The two routes should give the SAME value — that's the framework's discipline.  If Route H gives one answer and Route C gives another, the closure isn't structural (something is missing).

For OPEN hypotheses (where routes don't close), document the candidate $c_g$ value, the suggestive numerical pattern, and the candidate routes — at LAYER-1 HYPOTHESIS grade.  Don't graduate to theorem-grade until BOTH routes close to the same value.

### Step 5 — Calibration check

Verify that the proposed mechanism reproduces $c_v = 5/12$ for v_Higgs via the same machinery.  If your derivation gives the right $c_g$ for your observable but doesn't also give $5/12$ for v_Higgs, it's not the framework's mechanism.

### Step 6 — Numerical validation (NOT the criterion)

Compute the predicted $g_{\rm physical}$ and compare to PDG/observation.  Match is INDICATIVE but not the test of correctness — structural derivation per Steps 4–5 IS the test, per `feedback_no_post_hoc_structural_backfill.md`.

### Step 7 — Grade declaration

| Steps closed | Grade |
|---|---|
| Steps 1–6 all closed; Routes H + C both derive same $c$; calibration check passes | THEOREM-GRADE |
| Steps 1–3 + Step 4 partial (one route closes but not both); numerical match good | A− or hypothesis-grade |
| Step 4 open (no route closes); numerical pattern suggestive; structural form is clean rational | Layer-1 hypothesis |
| Numerical match but no structural derivation; form is unclean rational | numerological — DO NOT graduate |

---

## 7. Theory references

### Mechanism docs (under `docs/theorems/`)

- `theorem_dark_correction_mdl.md` — MDL ranking of parity-odd functionals (Lemma 1); photon Berry phase (Lemma 2); linear-vs-squared rule (Lemma 3, conditional).
- `theorem_m_nu_dark_correction_uniqueness_closure.md` — Feshbach contour integral uniqueness for $m_\nu$ (Im/h forced by mechanism).
- `theorem_dark_5_12_spectral.md` — 5/12 via Hashimoto-spectral marginal-mode fraction.
- `theorem_unified_spectral_dark.md` — unified spectral picture (Class A master theorem).
- `theorem_analytical_feshbach_ramanujan_boundary.md` — Feshbach contour boundary analytics.
- `theorem_ifeshbach_percycle_resolution.md` — per-cycle Feshbach resolution.
- `theorem_dark_map_class2_closure.md` — Class 2 dark-sector map closure.

### Cluster characterization (under an internal working note)

- `substrate_feshbach_analog_cluster_2026-05-14.md` — cluster master, sharpened analytic structure.
- `theorem_dark_correction_taxonomy_scoping.md` — older 4-pathway taxonomy (superseded by MDL synthesis).
- `alpha_GUT_dark_correction_verdict_2026-05-14.md` — α_GUT hypothesis (this work).
- `P4_joint_feshbach_y_tau_2026-05-09.md` — y_τ joint Feshbach scoping.
- `Lambda_CC_path_B_w_eff_mixing_scoping_2026-05-05.md` — Λ_CC Path B V_Ram split.

### Substrate-level supporting work

- `predictions/uniform_Q_density_derivation.md` — uniform Q-space density on Ramanujan circle (theorem-grade Part A).
- `predictions/alpha_1.py` — α_1_bare = (2/3)^(g-2) derivation.
- `proofs/foundations/q_space_dark_correction_unification.py` — Q-space spectrum verification.
- `proofs/foundations/dark_feshbach_a2_closure.py` — 5/12 dual verification.

### Mass mechanism (foundational)


### Axioms

- `docs/framework/framework_axioms.md` §5b — A5(b): MDL probability = coupling.
- `docs/theorems/theorem_A2_mdl_from_finite_register.md` — A2-T waterline (selective retention).

---

## 8. Discipline: protections against numerology

The framework's dark-correction protocol has explicit safeguards against post-hoc structural backfill (per an internal note and `feedback_audit_for_smuggled_parameters_2026-05-14.md`):

1. **TWO derivation routes per coefficient.** Theorem-grade requires Routes H AND C to give the same number.  A coefficient derived from one route alone is at most A−.

2. **Calibration via v_Higgs.** Any derivation mechanism for a new $c_g$ must also reproduce $c_v = 5/12$.  No exceptions.

3. **Discovery-order discipline.** Document whether the structural derivation was completed BEFORE or AFTER the numerical match was known.  A "structural reading" constructed post-hoc to match data is numerology, not derivation.

4. **Clean-rational test.** $c_g$ must be expressible in $K = \mathbb{Q}(\sqrt 2, \sqrt 3, \sqrt 5)$ via framework primitives.  $c_g \approx 0.148$ with no clean form is suggestive of an open mechanism, not a closure.

5. **Alternative-form discrimination.** When multiple structural forms give similar numerics ($c = N_{\rm atoms}/(2|E|) = 1/6$ vs $c = N_{\rm directed}/(N_{\rm atoms} \cdot k_*^2) = 1/3$ for α_GUT), the choice between them must come from STRUCTURE (which mechanism applies), not from which fits data better.

6. **Children-propagation gate.** Hypothesis-grade $c_g$ values are NOT propagated to children predictions until graduation.  The cluster predictions inherit only theorem-grade dark corrections.

---

## 9. Open structural questions

The framework has the universal template but doesn't yet have a derivation mechanism that works for every observable class.  Open work:

(O1) **$c_\lambda$ for Higgs quartic** ($\lambda$) — **CLOSED 2026-05-15 EOD.** Routes H + C closed for c_H at theorem-grade. Family D per-leg multiway dark-disruption (§3 (D)) gives $\delta\lambda/\lambda = -4\alpha_1^2 \approx -0.609\%$ from the 4-Higgs-leg |φ|⁴ vertex. Predicted m_H = 125.195 GeV vs observed 125.20 GeV (**−0.05σ_PDG** vs prior tree-level +3.43σ_PDG). See `family_D_route_H_2026-05-15.py` and `family_D_route_C_2026-05-15.py`.

(O2) **$c_{y_\tau}$ for Yukawa vertex** — **CLOSED 2026-05-15 EOD.** Routes F-1 + F-2 closed for c_F. Family D gives $\delta y_\tau/y_\tau = -(5/6)\alpha_1^2 \approx -0.127\%$ from the 1H+2F Yukawa vertex × c_H (Routes H+C) + c_F (Routes F-1+F-2). Predicted m_τ = 1.7768 GeV vs observed 1.77686 GeV (**−0.17σ_PDG** vs prior tree-level +18.67σ_PDG). See `family_D_route_F_2026-05-15.py` and `family_D_route_F2_2026-05-15.py`.

(O1+O2 joint) **m_H +3.43σ_PDG residual decomposition.** Under Family D, the residual splits as (4 H-leg dark) + (Bose factor of √(2λ)) on m_H, giving total $-2\alpha_1^2 \approx -0.305\%$ on m_H. Predicted m_H closes from 125.578 to 125.195 GeV (−0.05σ_PDG vs observed 125.20). m_τ from 1.7791 to 1.7768 GeV (−0.17σ_PDG vs observed 1.77686). Whole Yukawa+Higgs Clause 8 sector graduates to sub-σ_PDG match if Family D graduates. See `proofs/foundations/dark_disruption_per_leg_2026-05-15.py` for sentinel verification.

(O3) **$c_{\alpha_{\rm GUT}} = 1/k_*$** — **CLOSED 2026-05-15 EOD+1.** Routes H + C both closed in commit `f481dbd` (`theorem_alpha_GUT_dark_correction.md` § 3-4); observable-class selection rule substrate-derived in commit pending (`proofs/foundations/alpha_GUT_selection_rule_substrate_derivation.py`).  The Type 3 Peskin-Schroeder / Weinberg imports for the gauge-singlet-exclusion rule are retired in favor of Type 4 inheritance from `theorem_h1_master_compression.md` + Type 3 Wilson 1974 / Bass-Stark-Terras — both substrate-aligned, both already used elsewhere in the framework.  Graduates THEOREM-GRADE-CONDITIONAL → **THEOREM-GRADE** on substrate-aligned conditions only.  Cluster (P63–P71) inherits unchanged numerical values; provenance fully substrate-aligned.

(O4) **β photon coefficient $c = 1$** — **CLOSED 2026-04-29.**  Theorem-grade via the uniqueness template (D1 substrate-chirality + MDL Lemma 1 sin(arg h) parity-odd selection + D2 algebraicity meta-theorem ruling out 1/(16π²) by Lindemann 1882).  See `theorem_beta_uniqueness_closure.md` + `theorem_lattice_coupling_algebraicity.md`; parameter ledger Row P44 graduated UNIQUE-THEOREM-GRADE (commit 3aaa473).  Predicted β = 0.331° vs Eskilt 2022 0.342° ± 0.094° → +0.12σ.

(O5) **Λ_CC Path B V_Ram split** — distinct mechanism (sub-sector projection); doesn't fit the universal template.

(O6) **Multi-loop / higher-order corrections.** All current closures are single-loop / one-Σ_Q-insertion.  Higher-order would shift $c_g$ by $O(\alpha_1^2)$ — sub-percent on observable.  Not load-bearing unless precision improves.

(O7) **M_Z scale shift / $\delta_r$** (added 2026-05-15 EOD+1; **RESOLVED 2026-05-16**).  The sign-uniform absolute-$M_Z$ oblique is the Z/Perron eigen-channel of the unified-oblique resolvent: $\delta_r = c_S\,\alpha_1/(1-\alpha_1)$ with $c_S = 1/(2|E|) = 1/12$ DERIVED as the $B_{NB}$ Perron-residue gauge-singlet projection (Route H ≡ Route C by handshake lemma $2|E|=Nk^*$).  The earlier $c_{M_Z}\approx 0.175$ Family-B fit and the retracted Phase-A citation are SUPERSEDED.  `theorem_unified_oblique.md`; `predictions/delta_r.py` (Row P64).

(O8) **ρ-parameter shift $\delta\rho \approx +1.05\%$** (added 2026-05-15 EOD+1; **RESOLVED 2026-05-15 EOD+16 (Phase C/C.1), unified 2026-05-16**).  Closed as a SINGLE Hashimoto spectral object $\delta\rho = \tfrac12\cdot\tfrac{\sqrt5}{4}\cdot(2/3)^8 = +1.091\%$ ($+0.76\sigma_{\rm obs}$): the W/$h_P$ eigen-channel of the unified-oblique resolvent (Family-E mass²-Feshbach).  The $3/(32\pi^2)$ A4 / $y_t(\rm GUT)=1$ candidate and its open sub-items (a)–(c) are **REJECTED/SUPERSEDED** — $1/(32\pi^2)\notin K$ (O9 algebraicity violation); no QFT-loop-normalization analog is needed.  `theorem_unified_oblique.md`; `predictions/delta_rho.py` (Row P73).

(O9) **Substrate loop normalization** (added 2026-05-15 EOD+1, foundational).  Family D succeeded for vertex-level corrections by per-leg counting.  But propagator-level corrections (Δρ, gauge-boson self-energy, fermion self-energy at the propagator) need a *substrate analog of QFT 1-loop normalization* — the factor $1/(16\pi^2)$ or $1/(32\pi^2)$ that appears in continuum QFT loops.  This is a foundational structural object the framework needs but doesn't yet have.  Connection: NA-4 multiway formalism + Hashimoto operator spectrum + per-cycle Feshbach (`theorem_ifeshbach_percycle_resolution.md`) should be the natural domain.  Open since 2026-05-15.

---

## 10. Summary for parameter derivers

When deriving a parameter, ask:

1. **Tensor character + family fit?** (Step 1 + Step 1b — vertex vs propagator
   per §4.5 meta-classification.)
2. **Algebraicity?** (Step 1c — closure form in K = ℚ(√2, √3, √5); π-audit
   on any Type-3 SM mechanism citation per parameter_linter.md Clause 9.)
3. **Mechanism?** (Berry / Feshbach / counting / per-leg / block / sub-sector;
   or Family E if custodial-breaking.)
4. **What $c_g$ does it predict?** (Apply Route H or Route C per Step 4.)
5. **Does it pass calibration?** (Reproduces v's 5/12.)
6. **Does it pass clean-rational test?** ($c_g$ in $K$.)

If all six steps close: THEOREM-GRADE dark correction.  Propagate to children.

If steps 1–3 close but 4 fails: open hypothesis.  Record but don't propagate.

If a numerical match works without structural mechanism: NUMEROLOGY.  Don't ship.

If steps 1–2 reveal a propagator-level custodial-breaking observable
(M_Z/m_W ρ-shift type): tag as STRUCTURAL-DERIVATION-CONDITIONAL with
named Family-E open gap.  Don't apply Family A/B/C/D to force closure.

If steps 1c reveals a SM-mechanism citation imports continuum loop
factors (1/(16π²), Sirlin Δr, Δα_had): the citation is K-INVALID
(Clause 9 violation).  Either derive K-rational substrate analog
(legitimate closure) or tag bridge-convention-only.

---

## 11. Reading order

For a parameter deriver new to the framework's dark-correction mechanism:

1. This master doc (overview).  Focus on §3 (family taxonomy), §4.5
   (vertex/propagator meta-classification), §6 (application protocol).
2. `theorem_dark_correction_mdl.md` (Lemma 1 + Lemma 2 mechanics).
3. `theorem_m_nu_dark_correction_uniqueness_closure.md` (Feshbach example).
4. `theorem_dark_5_12_spectral.md` (5/12 spectral derivation).
5. `proofs/foundations/dark_feshbach_a2_closure.py` (5/12 cycle-counting derivation).
6. `substrate_feshbach_analog_cluster_2026-05-14.md` (cluster characterization).
7. The relevant observable-class theorem (m_ν uniqueness for masses, etc.).
8. **`theorem_lattice_coupling_broader_implications.md`** (algebraicity
   meta-theorem; K-rational filter + positive search procedure).
9. **`parameter_linter.md` Clause 9** (Type-3 SM import π-audit).
10. **`vertex_propagator_audit_open_observables_2026-05-15.md`**
    (audit table of open observables by tensor character).

Then apply the protocol in §6 to your observable.
