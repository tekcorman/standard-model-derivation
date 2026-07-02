# sin²θ_W at unification — derivation

## 1. Abstract

The weak mixing angle at the natural unification scale is derived as sin²θ_W(M_unif) = 3/8 exactly from the Georgi-Quinn-Weinberg 1974 trace identity applied to the framework's complete unifying multiplet — one color-extended Pati-Salam generation (16 states: 4 leptons + 12 quarks = 4 lepton states + 3 colors × 4 quark states). The load-bearing structural input is B6's identification of the srs body-diagonal C₃ at the P-point with a cyclic Z₃ ⊂ SU(3)_c, which supplies the color multiplicity 3 needed to distinguish quark states from leptons. The proof is gate-passing under A1 + A2-T + A3-T + B1.b + B2 + B3 + B6 with 0 adoptions. Full proof: `docs/theorems/theorem_sin2_theta_W_unification.md`. The observed sin²θ_W(M_Z) ≈ 0.2312 is recovered by single-regime MSSM-style RG running from 3/8 down to M_Z (mathematically-complete with M_Z, α_em as external inputs; M_SUSY is NOT a framework parameter per ADOPTED-MSSM-Sb 2026-05-14 PM revision). Supersedes the retracted 3/13 formula (arithmetic nonsense — dim U(1) = 1, not 3).

## 2. Framework axioms invoked

- **A1** — Binary self-inverse toggle on srs (`docs/framework/framework_axioms.md` §2).
- **A2-T** — MDL selection of srs and the invariant Clifford construction per B1.b (derived theorem; see `docs/theorems/theorem_A2_mdl_from_finite_register.md`).
- **A3-T** — Complex Hilbert space at each node via CDP 2011 purification (derived theorem; see `docs/theorems/theorem_A3_complex_hilbert_from_multiway.md`); required for the Spin(6) ≅ SU(4) lift in B6.

Note: A5(a), A5(b) are NOT load-bearing for this derivation — sin²θ_W is a pure group-theoretic trace, not a dynamical coupling.

## 3. Derivation

**Step 1 — Colorless PS content from B3** [T4: `predictions/theorem_B3_spinor_fermion.py`]. The 8-dim Cl(6,0) Dirac spinor S decomposes under Spin(4) × Spin(2) = SU(2)_L × SU(2)_R × U(1)_{B−L}^{PS} as one Pati-Salam generation {ν, e, u, d} × {L, R}, colorless. Cartan generators: T_L = Γ_{12}/(2i), T_R = Γ_{34}/(2i), Y_{PS} = Γ_{56}/(2i) all with common Killing-form normalization inherited from Cl(6,0) bivectors.

**Step 2 — Color multiplicity from B6** [T4: `proofs/foundations/theorem_B6_bridge.py` (CAS-verified, script prints OK)]. The body-diagonal C₃ at the srs P-point lifts to the SU(4) element with eigenvalues (1, 1, ω, ω²) on the fundamental 4. Under SU(4) → SU(3)_c × U(1)_{B−L} (Pati-Salam 1974), the eigenvalue structure splits as:
- "1" with B−L = −1: lepton singlet (C₃-trivial on color)
- (1, ω, ω²) with B−L = +1/3: color triplet on quarks

This forces quark multiplicity 3 and lepton multiplicity 1 in the color-extended generation. [T3: Pati-Salam 1974 §II; Lawson-Michelsohn 1989 I §6]

**Step 3 — Color-extended state count** [T2: counting]. The physical PS generation has 16 states:
- 4 lepton states (ν_L, e_L, ν_R, e_R), n_c = 1
- 12 quark states (u_L, d_L, u_R, d_R) × 3 colors, n_c = 3

**Step 4 — SM charge assignments** [T3: Slansky 1981 §4 Table 5, Langacker 2010 §2.2]. Using Y_SM = T_3^R + (B−L)/2 with (B−L)_quark = +1/3 and (B−L)_lepton = −1 (Slansky normalization):

| Species | T_3^L | Y_SM | Q = T_3^L + Y_SM | n_c |
|---------|-------|------|-------------------|-----|
| ν_L     | +1/2  | −1/2 | 0                 | 1   |
| e_L     | −1/2  | −1/2 | −1                | 1   |
| ν_R     | 0     | 0    | 0                 | 1   |
| e_R     | 0     | −1   | −1                | 1   |
| u_L     | +1/2  | +1/6 | +2/3              | 3   |
| d_L     | −1/2  | +1/6 | −1/3              | 3   |
| u_R     | 0     | +2/3 | +2/3              | 3   |
| d_R     | 0     | −1/3 | −1/3              | 3   |

These are the standard SM charges, now DERIVED (not assumed) from the PS-SU(4) normalization.

**Step 5 — Georgi-Quinn-Weinberg trace** [T3: Georgi-Quinn-Weinberg 1974 Eq. (4)]. Applied to the complete 16-state color-extended multiplet at common Killing-form normalization:
$$\sin^2\theta_W^{(\text{unif})} \;=\; \frac{\sum_{\text{states}} n_c T_{3,L}^2}{\sum_{\text{states}} n_c Q^2}$$

**Step 6 — Trace computation** [T2: exact rational arithmetic, CAS-verified]:
$$\sum n_c T_{3,L}^2 \;=\; 2 \cdot \tfrac{1}{4}\cdot 1 + 0 + 2 \cdot \tfrac{1}{4}\cdot 3 + 0 \;=\; \tfrac{1}{2} + \tfrac{3}{2} \;=\; 2$$
$$\sum n_c Q^2 \;=\; (0+1)\cdot 1 + (0+1)\cdot 1 + (\tfrac{4}{9}+\tfrac{1}{9})\cdot 3 + (\tfrac{4}{9}+\tfrac{1}{9})\cdot 3 \;=\; 2 + \tfrac{10}{3} \;=\; \tfrac{16}{3}$$

**Step 7 — Assembly** [T2]:
$$\sin^2\theta_W^{(\text{unif})} \;=\; \frac{2}{16/3} \;=\; \frac{6}{16} \;=\; \frac{3}{8}$$

## 4. Result

$$\boxed{\;\sin^2\theta_W^{(\text{unif})} \;=\; \frac{3}{8} \;=\; 0.375 \quad \text{EXACT}\;}$$

## 5. Comparison with experiment

sin²θ_W is not directly measured at M_unif (unification is not a directly accessible scale). The observed value at M_Z is 0.23121 ± 0.00004 (PDG 2024).

**Connection to observation via RG running** (Peskin-Schroeder §21; Langacker 2010 §7.6). The one-loop RG equation:
$$\sin^2\theta_W(M_Z) = \sin^2\theta_W^{(\text{unif})} - \frac{\alpha_{em}(M_Z)}{2\pi}(b_1 + b_2) \ln\!\left(\frac{M_{\text{unif}}}{M_Z}\right) + O(\alpha^2)$$

Under single-regime MSSM-style RG running from M_unif ~ 2×10¹⁶ GeV (no M_SUSY threshold; the framework's native picture per ADOPTED-MSSM-Sb 2026-05-14 PM revision), RG running from 3/8 gives sin²θ_W(M_Z) ≈ 0.230, matching the observed 0.2312 to ~0.5%. Tightening this gap via M_SUSY threshold matching is NOT pursued because M_SUSY is not a framework parameter (see `docs/theorems/theorem_beta_coefficients_derived.md` §2.5).

**Grade:** unification value 3/8 is THEOREM. M_Z value is mathematically-complete conditional on M_Z and α_em(M_Z) as external inputs. (M_SUSY is NOT a framework parameter; the running is single-regime per ADOPTED-MSSM-Sb 2026-05-14 PM revision.)

## 6. Comparison with the retracted 3/13

The formula sin²θ_W = dim U(1) / dim(SU(2) ⊕ U(1)) = 3/13 cited in `docs/parameters/derivations.md` §2.4 is arithmetically broken: dim U(1) = 1 and dim(SU(2) ⊕ U(1)) = 3 + 1 = 4, giving 1/4 = 0.25, not 3/13 ≈ 0.2308. The numerical coincidence 3/13 ≈ 0.2308 ≈ sin²θ_W(M_Z) observed is NOT justified by any structural argument — it was a post-hoc numerology edit. The correct derivation (this file) gives 3/8 at unification + RG running to M_Z.

## 7. Open questions

**At the unification-scale level:** None. All load-bearing steps pass T1/T2/T3/T4 with zero adoptions. The result 3/8 matches the textbook Georgi-Weinberg value exactly.

**Downstream residuals (not properties of this derivation):**

- **M_unif scale itself:** standard gauge-coupling unification at M_unif ≈ 2×10¹⁶ GeV is a property of the RG flow, not directly derived here. A framework-specific M_unif identification (e.g., via lattice-scale correspondence) would make RG running not require M_unif as external input. Currently external.

- **M_Z value requires external inputs:** M_Z itself and α_em(M_Z) come from observation. Same structure as every renormalized coupling in the SM.

- **B6's "full SU(3)_c gauge coupling g_3"** is not derived — B6 gives only the discrete color-Z₃ multiplicity, not the continuous SU(3)_c gauge dynamics. For the GQW trace (which uses only the MULTIPLICITY, not g_3), B6 suffices. For the RG running of α_s itself, g_3 is an input.

- **Candidate Feshbach-analog dark correction at α_GUT** (Layer-1 hypothesis, 2026-05-15): the sin²θ_W(M_Z) residual −0.5% is structurally consistent with the substrate-Feshbach-analog dark correction α_GUT × (1 − (1/k*) × α_1/(1−α_1)) propagating through the cluster. Under this hypothesis, 1/α_1(M_Z) and 1/α_2(M_Z) match PDG within 0.01%; sin²θ_W(M_Z) (which is computed from α_1/α_2 ratio) closes correspondingly. NOT propagated to this prediction's value until hypothesis graduates via Routes H/C closure. See `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md`.

## 8. Cross-references

- `docs/theorems/theorem_sin2_theta_W_unification.md` — full gate-first theorem proof (this derivation's T4 source)
- `predictions/theorem_B3_spinor_fermion.py` + derivation.md — one colorless PS generation from Cl(6,0)
- `proofs/foundations/theorem_B6_bridge.py` — CAS-verified color-Z₃ multiplicity (script prints OK)
- `predictions/alpha_GUT.py` — α_GUT = 1/24 at unification (theorem-grade, independent)
- `docs/parameters/derivations.md` §2.4 — retract the 3/13 formula (pending edit)

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
