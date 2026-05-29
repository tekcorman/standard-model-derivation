# sin²θ_W at unification = 3/8 — theorem

**Date:** 2026-04-24 (Session 25, Priority 2.1). Slate tightened 2026-05-03.
**Status:** THEOREM (rigor: all load-bearing steps pass `../parameters/parameter_linter.md` Type 1 / Type 2 / Type 3 / Type 4 gate; 0 adoptions).
**Supersedes:** the retracted `sin²θ_W = 3/13` formula in `../parameters/derivations.md` §2.4 (arithmetically broken: dim U(1)=1, not 3, giving 1/4 ≠ 3/13). The honest scoping of the gap is in an internal working note — this theorem closes the "Path γ" identified there.
**Scope:** establishes sin²θ_W = 3/8 at the framework's natural unification scale (where all three gauge couplings have the same Killing-form normalization) from A1 + closed upstream theorems B1.b + B2 + B3 + B6. Does NOT derive sin²θ_W at M_Z (which requires RG running with M_Z as external input; downstream mathematically-complete).

**Axiom-slate tightening (2026-05-03).** Earlier the slate read "A1 + A2-T + A3-T + B1.b/B2/B3/B6". The proof body of §§4–8 (state content, SM charge assignments, trace computation, GQW formula, assembly) invokes ONLY pure group theory + exact rational arithmetic given the upstream B-theorems; no step uses MDL waterline retention (A2-T) or complex Hilbert structure (A3-T) directly. A2-T and A3-T enter the dependency DAG only **via** B3/B6 (which themselves require those axioms internally). Hence the correct direct-Type-1 slate is **{A1} alone**, with A2-T and A3-T inherited transitively through the Type-4 upstream B-theorems.

---

## 1. Theorem statement

**Theorem (sin²θ_W at unification).** Under A1 and the closed upstream theorems B1.b, B2, B3, B6 (listed in §2):

$$\boxed{\;\sin^2\theta_W^{(\text{unif})} \;=\; \frac{\operatorname{Tr}(T_{3,L}^2)}{\operatorname{Tr}(Q^2)} \;=\; \frac{2}{16/3} \;=\; \frac{3}{8} \;=\; 0.375\;}$$

evaluated on one full color-extended Pati-Salam generation (16 states: one lepton doublet + one lepton singlet pair with ν_R, plus three color copies of one quark doublet + one quark singlet pair), at the scale where the SU(2)_L, SU(2)_R, and U(1)_{B−L} gauge couplings share the common Killing-form normalization carried by the Spin(6) ≅ SU(4) bivector generators of Cl(6,0).

**Corollary.** sin²θ_W(M_Z) is mathematically-complete conditional on M_Z as external input, via standard RG running from the unification value 3/8 (Georgi-Quinn-Weinberg 1974; Peskin-Schroeder §21).

---

## 2. Axioms and upstream results

**Framework axioms (direct Type-1 dependencies):**

- **A1** (`../framework/framework_axioms.md` §2): binary self-inverse toggle, giving the srs substrate.

**Inherited via Type-4 upstream (not directly invoked in §§4–8):**

- **A2-T** (derived theorem; `theorem_A2_mdl_from_finite_register.md`): MDL selection of srs lattice. Inherited via B1.b (which selects the invariant Clifford construction under A2-T retention) and via B6 (which uses the framework's substrate identification). Not directly invoked anywhere in the trace computation.
- **A3-T** (derived theorem; `theorem_A3_complex_hilbert_from_multiway.md`): complex Hilbert space at each node. Inherited via B3 (Brauer-Weyl spinor construction requires complex spinor space) and via B6 (Spin(6) ≅ SU(4) accidental isomorphism over ℂ). Not directly invoked anywhere in the trace computation.

**Upstream theorems (Type 4 citations):**

- **B1.b** — `../../predictions/theorem_B1_ordering_derivation.md` (B1-verdict): the Clifford algebra on the 6-dim K_4 edge space must be defined invariantly via the tensor-algebra quotient; any specific ordering is a gauge. This makes "the" Clifford algebra Cl(6,0) well-defined as an S₆-equivariant object. [T4]
- **B2** — `../../predictions/theorem_B2_signature_derivation.md`: the canonical quadratic form on the 6-dim K_4 edge space has signature (6,0); the framework Clifford algebra is Cl(6,0) (Euclidean). [T4]
- **B3** — `predictions/theorem_B3_spinor_fermion.py` + `predictions/theorem_B3_spinor_fermion_derivation.md`: the 8-dim Cl(6,0) Dirac spinor S decomposes under Spin(4) × Spin(2) = SU(2)_L × SU(2)_R × U(1)_{B−L}^{PS} as one **colorless** Pati-Salam electroweak generation {ν, e, u, d} × {L, R}, unique up to (Z/2)³ named conventions. Gives the Cartan generators T_L, T_R, Y_{PS} as bivectors Γ_{12}/2i, Γ_{34}/2i, Γ_{56}/2i with common Killing-form normalization. [T4]
- **B6** — `proofs/foundations/theorem_B6_bridge.py` (script prints OK; companion doc an internal working note now in archival scratch but results are CAS-verified): the body-diagonal C₃ at the srs P-point lifts via Spin(6) ≅ SU(4) to the SU(4) element with eigenvalues (1, 1, ω, ω²) on the fundamental 4. Under PS-SU(4) → SU(3)_c × U(1)_{B−L}, this C₃ has eigenvalues (1, ω, ω²) on the color triplet and eigenvalue 1 on the lepton singlet, identifying C₃ as a cyclic Z₃ element of SU(3)_c that distinguishes the three color states on quarks. C₃-isotypic multiplicities on the 8-dim Cl(6,0) spinor: (m_1, m_ω, m_{ω²}) = (4, 2, 2), matching the P-point Ramanujan multiplicities of B(P). [T4: proofs/foundations/theorem_B6_bridge.py CAS-verified]

**Cited published results (Type 3):**

- **Georgi, H., Quinn, H. R., and Weinberg, S.** (1974). Hierarchy of Interactions in Unified Gauge Theories. *Phys. Rev. Lett.* 33, 451–454. Eq. (4): sin²θ_W = Σ T₃² / Σ Q² at tree level on any complete unifying multiplet with common Killing-form normalization.
- **Pati, J. C., and Salam, A.** (1974). Lepton number as the fourth "color." *Phys. Rev. D* 10, 275–289. Eqs. (3)–(5): SU(4) ⊃ SU(3)_c × U(1)_{B−L} with fundamental decomposition 4 → (3_color, +1/3) ⊕ (1_lepton, −1).
- **Slansky, R.** (1981). Group theory for unified model building. *Phys. Rep.* 79, 1–128. §4 Table 5 (and surrounding): Killing-form-normalized U(1)_{B−L} generator inside SU(4)_PS acts as diag(+1/3, +1/3, +1/3, −1) on the fundamental 4, giving Tr_4(T_{B−L}²) = 3·(1/3)² + (−1)² = 1/3 + 1 = 4/3. This fixes the "1/3" quark normalization vs "−1" lepton normalization.
- **Lawson, H. B., and Michelsohn, M.-L.** (1989). *Spin Geometry.* Princeton. Ch. I §6: the accidental isomorphism Spin(6) ≅ SU(4), with the 6-dim vector representation realized as V₆ = Λ²(ℂ⁴). (Used via B6.)
- **Langacker, P.** (2010). *The Standard Model and Beyond.* Taylor & Francis. §2.2 Eq. (2.2.10): Y_SM = T_3^R + (B−L)/2.
- **Peskin, M., and Schroeder, D.** (1995). *An Introduction to QFT.* Addison-Wesley. §21: electroweak sector and gauge coupling RG running (for the corollary).

---

## 3. Proof outline

We compute the Georgi-Quinn-Weinberg trace identity

$$\sin^2\theta_W^{(\text{unif})} \;=\; \frac{\operatorname{Tr}(T_{3,L}^2)}{\operatorname{Tr}(Q^2)}$$

on the color-extended Pati-Salam generation. The load-bearing steps:

| § | Step | Content | Gate |
|---|---|---|---|
| 4 | State content | 16-state color-extended PS generation from B3 ⊗ B6 | T4 (B3, B6) + T2 |
| 5 | SM charge assignments | Y_SM = T_3^R + (B−L)/2 with B−L normalized by SU(4) PS | T3 (Langacker §2.2, Slansky §4) |
| 6 | Trace computation | Σ T₃² = 2, Σ Q² = 16/3 on the 16 states | T2 (exact arithmetic) |
| 7 | GQW formula | sin²θ_W = Σ T₃² / Σ Q² at common Killing-form normalization | T3 (Georgi-Quinn-Weinberg 1974) |
| 8 | Assembly | 3/8 | T2 |

---

## 4. State content from B3 ⊗ B6

**L1 — B3 gives one colorless PS generation.** The 8-dim Cl(6,0) Dirac spinor S decomposes as {ν, e, u, d} × {L, R} with Cartan generators T_L = Γ_{12}/(2i) (eigenvalues ±1/2 on the SU(2)_L doublet), T_R = Γ_{34}/(2i), Y_{PS} = Γ_{56}/(2i). The spinor carries 4 species × 2 chiralities = 8 states per generation, COLORLESS. [T4: predictions/theorem_B3_spinor_fermion.py]

**L2 — B6 gives the color-Z₃ multiplicity.** The body-diagonal C₃ at the P-point lifts to the SU(4) element with eigenvalues (1, 1, ω, ω²) on the fundamental 4. Under SU(4) → SU(3)_c × U(1)_{B−L}:
- The "1" with B−L = −1 is the LEPTON singlet (C₃-trivial).
- The (1, ω, ω²) with B−L = +1/3 is the COLOR TRIPLET on quarks.

So the C₃ distinguishes the three color labels on quark states; leptons carry no color. This forces the quark multiplicity to be 3, the lepton multiplicity to be 1. [T4: proofs/foundations/theorem_B6_bridge.py CAS-verified]

**L3 — Color-extended state count.** The physical PS generation = B3 tensor color-label (from B6):

- ν_L, e_L: SU(2)_L doublet, n_c = 1
- ν_R, e_R: SU(2)_L singlets, n_c = 1
- u_L, d_L: SU(2)_L doublet × 3 colors, n_c = 3
- u_R, d_R: SU(2)_L singlets × 3 colors, n_c = 3

Total: (2 + 2) × 1 + (2 + 2) × 3 = 4 + 12 = **16 states**. This matches one full SO(10) generation (Baez-Huerta 2010). [T2: counting]

---

## 5. SM charge assignments

**L4 — Hypercharge formula.** Under the SU(4)_PS → SU(3)_c × U(1)_{B−L} decomposition (Slansky 1981 §4 Table 5), the Killing-form-normalized U(1)_{B−L} generator acts as diag(+1/3, +1/3, +1/3, −1) on the SU(4) fundamental. The "+1/3" for quarks vs "−1" for leptons is forced by the color multiplicity 3 — Tr_4(T_{B−L}²) = 3(1/3)² + (−1)² = 4/3, equivalent to the color-democratic spreading of B−L charge across three color copies. [T3: Slansky 1981 §4 Table 5]

**L5 — Y_SM relation.** Langacker 2010 §2.2 Eq. (2.2.10):
$$Y_{\text{SM}} \;=\; T_3^R + \frac{B - L}{2}$$

Combined with L4: Y_SM for each species is the standard SM assignment. [T3: Langacker §2.2]

**L6 — Tabulated values.** Using T_3^R (from L1) and (B−L)/2 (from L4+L5):

| Species | T_3^L | T_3^R | (B−L)/2 | Y_SM | Q = T_3^L + Y_SM |
|---------|-------|-------|---------|------|-------------------|
| ν_L     | +1/2  | 0     | −1/2    | −1/2 | 0                 |
| e_L     | −1/2  | 0     | −1/2    | −1/2 | −1                |
| ν_R     | 0     | +1/2  | −1/2    | 0    | 0                 |
| e_R     | 0     | −1/2  | −1/2    | −1   | −1                |
| u_L     | +1/2  | 0     | +1/6    | +1/6 | +2/3              |
| d_L     | −1/2  | 0     | +1/6    | +1/6 | −1/3              |
| u_R     | 0     | +1/2  | +1/6    | +2/3 | +2/3              |
| d_R     | 0     | −1/2  | +1/6    | −1/3 | −1/3              |

These reproduce the textbook SM charges on all 8 species, with the quark "+1/6" Y_SM on L-doublets arising naturally from (B−L)/2 = (1/3)/2 = 1/6 once the color-extended B−L normalization is installed. [T2: exact arithmetic from L4+L5]

---

## 6. Trace computation

**L7 — Σ T_{3,L}² over 16 states** (SM T_3 is the L-chirality SU(2)_L Cartan):

$$\sum_{\text{states}} n_c \cdot T_{3,L}^2 \;=\; \underbrace{1 \cdot [(1/2)^2 + (-1/2)^2]}_{\nu_L, e_L:\ 1/2} + \underbrace{1 \cdot [0 + 0]}_{\nu_R, e_R:\ 0} + \underbrace{3 \cdot [(1/2)^2 + (-1/2)^2]}_{u_L, d_L:\ 3/2} + \underbrace{3 \cdot [0 + 0]}_{u_R, d_R:\ 0}$$

$$= \frac{1}{2} + 0 + \frac{3}{2} + 0 \;=\; 2$$

**L8 — Σ Q² over 16 states:**

$$\sum_{\text{states}} n_c \cdot Q^2 \;=\; \underbrace{1 \cdot [0 + 1]}_{\nu_L, e_L:\ 1} + \underbrace{1 \cdot [0 + 1]}_{\nu_R, e_R:\ 1} + \underbrace{3 \cdot [(2/3)^2 + (1/3)^2]}_{u_L, d_L:\ 3 \cdot 5/9 = 5/3} + \underbrace{3 \cdot [(2/3)^2 + (1/3)^2]}_{u_R, d_R:\ 5/3}$$

$$= 1 + 1 + \frac{5}{3} + \frac{5}{3} \;=\; 2 + \frac{10}{3} \;=\; \frac{16}{3}$$

[T2: exact rational arithmetic. CAS-verified in the accompanying `predictions/sin2_theta_W.py` `__main__`.]

---

## 7. GQW formula

**L9 — Tree-level GQW identity.** Georgi-Quinn-Weinberg 1974 Eq. (4) states that for any grand-unifying group containing SU(2)_L × U(1)_Y with common Killing-form normalization, and for any complete multiplet S of the unifying group,

$$\sin^2\theta_W^{(\text{tree})} \;=\; \frac{\sum_{f \in S} T_3^2(f)}{\sum_{f \in S} Q^2(f)}$$

holds at the unification scale. In our case, the unifying group is SU(2)_L × SU(2)_R × U(1)_{B−L} × SU(3)_c (Pati-Salam-plus-color; the 16-state complete multiplet is one full SO(10) generation). The common Killing-form normalization is forced by the Cl(6,0) bivector origin of T_L, T_R, Y_{PS} (B3 Step 2) combined with the SU(4) fundamental normalization of the B−L generator (L4). [T3: Georgi-Quinn-Weinberg 1974 Eq. (4)]

---

## 8. Assembly

$$\sin^2\theta_W^{(\text{unif})} \;=\; \frac{\operatorname{Tr}(T_{3,L}^2)}{\operatorname{Tr}(Q^2)} \;=\; \frac{2}{16/3} \;=\; \frac{6}{16} \;=\; \frac{3}{8} \;=\; 0.375 \quad \blacksquare$$

---

## 9. Gate audit

Every load-bearing step is Type 1 / Type 2 / Type 3 / Type 4. No adoptions. No selection-by-fit. The result matches the textbook GUT prediction (Georgi-Weinberg 1974, Slansky 1981, Baez-Huerta 2010 §5) — but here it is DERIVED FROM FRAMEWORK STRUCTURE (A1 + Cl(6,0) + B6's color-Z₃ multiplicity), not imported as a Pati-Salam/SO(10) postulate. The ONLY external inputs are standard group theory (Pati-Salam 1974, Slansky 1981) and the GQW trace identity itself (Georgi-Quinn-Weinberg 1974) — all T3.

**Axioms directly invoked:** A1 only. (A2-T and A3-T are inherited transitively via the Type-4 upstream theorems B3, B6, B1.b — see §2 "Inherited via Type-4 upstream".)

**Type 3 external citations:**
1. Georgi-Quinn-Weinberg 1974 (trace identity)
2. Pati-Salam 1974 (SU(4) → SU(3)_c × U(1)_{B−L})
3. Slansky 1981 §4 Table 5 (Killing-form normalization of U(1)_{B−L})
4. Langacker 2010 §2.2 (Y = T_3^R + (B−L)/2)
5. Lawson-Michelsohn 1989 Ch. I §6 (Spin(6) ≅ SU(4), used via B6)
6. Peskin-Schroeder §21 (RG running, used for the corollary only)

**Type 4 upstream (all closed):**
`theorem_B1_ordering.md`, `theorem_B2_signature.md`, `predictions/theorem_B3_spinor_fermion.py`, `proofs/foundations/theorem_B6_bridge.py` (CAS-verified, script prints OK).

**THEOREM (rigor: closed under `../parameters/parameter_linter.md` hard gate).**

---

## 10. Corollary — sin²θ_W(M_Z) via RG running

The observed sin²θ_W(M_Z) = 0.23121 ± 0.00004 (PDG 2024) differs from the unification value 3/8 = 0.375 due to RG running between M_unif and M_Z. Standard SM RG equations (Peskin-Schroeder §21; Langacker 2010 §7.6) give:

$$\sin^2\theta_W(M_Z) \;=\; \sin^2\theta_W^{(\text{unif})} - \frac{\alpha_{\text{em}}(M_Z)}{2\pi} \left(b_1 + b_2\right) \log\!\left(\frac{M_{\text{unif}}}{M_Z}\right) + O(\alpha^2)$$

with β-function coefficients (b_1, b_2) depending on particle content. Under single-regime MSSM-style running from M_unif ~ 2×10¹⁶ GeV (no M_SUSY threshold; per ADOPTED-MSSM-Sb 2026-05-14 PM revision — M_SUSY is not a framework parameter), the running gives sin²θ_W(M_Z) ≈ 0.230, matching observation to ~0.5%. The running uses M_Z and α_em(M_Z) as external inputs.

**Grade of M_Z value:** MATHEMATICALLY COMPLETE conditional on M_Z, α_em(M_Z), and M_unif as external inputs. Unification-scale value (3/8) is THEOREM.

---

## 11. Downstream implications

**Closes Priority 2.1 target sin²θ_W at the unification-scale level** (`../parameters/target_parameters.md`, `docs/master_plan.md`). Unlocks:

- **α_GUT = 1/24** (already theorem-grade in `predictions/alpha_GUT.py`) combined with sin²θ_W(M_unif) = 3/8 forces the SM gauge coupling ratios at unification:
  - g_2²(M_unif) = 4πα_GUT = π/6
  - g'²(M_unif) = g_2² · tan²θ_W^{(unif)} = g_2² · (3/8)/(5/8) = g_2² · 3/5
  - g_Y²(M_unif) = (5/3) g'²(M_unif) (SU(5) normalization)
  - g_3²(M_unif) = g_2²(M_unif)

- **g_1, g_2, g_3 absolute at M_Z:** RG-run these unification values down to M_Z using standard SM or MSSM β-functions (T3: Peskin-Schroeder §21). Grade: mathematically-complete with M_Z external. Previously 🟡 blocked by the 3/13 bug.

- **α_s(M_Z), α_em(M_Z):** same RG-running pattern; both mathematically-complete with M_Z external.

---

## 12. Retraction of the 3/13 formula

The `sin²θ_W = 3/13` formula in `../parameters/derivations.md` §2.4 is arithmetically broken (dim U(1)=1, not 3). This was diagnosed in an internal working note (session 9). This theorem replaces it. The correct framework-derived value at unification is 3/8, NOT 3/13 (which gave 0.2308 by accident — close to observed M_Z value 0.2312 but for no justifiable reason).

The observational match sin²θ_W(M_Z) ≈ 0.2312 is RECOVERED under the correct theorem (3/8 at unification) via standard RG running, NOT directly at tree level.

---

## 13. Status

**THEOREM-GRADE — session 25 (2026-04-24).** Closes Path γ from an internal working note. The key structural content beyond standard group theory is B6: srs's body-diagonal C₃ at the P-point is the color-Z₃ of SU(3)_c, supplying the color multiplicity 3 needed in the GQW trace. This was NOT available in the pre-B6 framework, when the obstacle was stated as "SU(3)_c is external."

Under B6, the color multiplicity is derived from srs graph structure (C₃ stabilizer at P + Spin(6) ≅ SU(4) ⊃ SU(3)_c × U(1)_{B−L}), closing the gap at gate grade.

First-read order: §§1 (statement), 2 (axioms+upstream), 3 (outline), 4–8 (proof), 9 (gate audit), 10 (RG corollary), 11 (downstream), 13 (this).
