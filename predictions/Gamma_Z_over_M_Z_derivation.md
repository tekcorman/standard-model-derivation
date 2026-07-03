# Γ_Z/M_Z — Derivation (F4 S3, 2026-07-02)

## 1. Abstract

We assemble the Z width fraction from the framework's derived electroweak endpoints and fermion
content through the tree-level golden rule dressed by the QCD series, with the assembly **frozen
before comparison**. Result: Γ_Z/M_Z = 0.027483 vs observed 0.0273634 ± 0.0000252 —
**+0.44%, +4.8σ_PDG: an OPEN residual**, prominently flagged and not relabeled. The residual's
located cause is the electroweak radiative layer (ρ_f, s̄²_eff, Z-bb̄ vertex), which the framework
has not derived — the same Layer-2 frontier as the M_Z oblique residual
(`theorem_M_Z_BZ_vacuum_polarization_2026-06-30.md`), which therefore now constrains the framework
through **two observables**. The pre-registered expectation band ("structure at ~0.5%, plausibly a
few σ_exp off") was stated before assembly and the result landed inside it.

## 2. Framework results invoked

Same as the companion `Gamma_W_over_Gamma_Z_derivation.md` §2, plus g₂(M_Z) = 0.65175
(`g_2.py`). The dark-cancellation lemma applies identically: the known dark sector cannot shift
Γ/M (real Perron dressing moves Γ and M together, exactly); no dark is applied — forbidden, not
omitted.

## 3. Derivation

**Step 1 (Type-3 structure, declared).** As companion §3 Step 1. Clause 9(9b) applies: 1/(48π)
is a continuum loop factor; grade capped at bridge-conditional.

**Step 2 (content and openness).** Cl(6)-Fock species read; top closed by the framework's own
m_t = 172.41 > M_Z/2. At s² = 0.23125: Σ_open = 7.3009, hadronic fraction f_had = 0.6913.

**Step 3 (assembly; algebra).**

$$\frac{\Gamma_Z}{M_Z} = \frac{g_2^2/c^2}{48\pi}\,\Sigma_{open}(s^2)\times
\Big[1 + f_{had}\Big(\frac{a_s}{\pi}+1.409\Big(\frac{a_s}{\pi}\Big)^2\Big)\Big].$$

M_Z's value does not enter (massless channels); G_F is unused; the ratio is therefore
non-circular against the width data.

**Step 4 (numerics).** Tree: $(0.65175^2/0.76875)\times 7.3009/(48\pi) = 0.026753$ (−2.23%).
QCD factor $1 + 0.6913\times(0.037529+0.001984) = 1.027317$ ⟹ **0.027483**.

## 4. Result

$$\boxed{\;\Gamma_Z/M_Z = 0.027483\;}$$

## 5. Comparison with experiment

Observed (PDG 2024): 0.0273634 ± 0.0000252 (±0.092%). Deviation: **+0.44%, +4.8σ_PDG.**
Clause 8 FAIL (8d) ⟹ grade **STRUCTURAL-DERIVATION-CONDITIONAL**: the structural assembly lands
within the pre-registered ±0.6% band on a 0.09%-measured rate observable, and the numerical match
is pending a named, un-derived mechanism. Per the top-down law the miss is OPEN — it is not an
artifact, not a floor, not grade-only.

## 6. Open questions

1. **The EW radiative layer** (the residual's located home): ρ_f, s̄²_eff, Z-bb̄ vertex — in SM
   language a −0.4%-scale effect with exactly the needed sign and size, but K-invalid to import
   numerically (Clause 9 bright line; Sirlin-Δr precedent). Deriving the substrate analog is the
   open equation; it is the SAME frontier as the M_Z +6.1σ oblique residual, now doubly
   constrained: any candidate derivation must move BOTH observables coherently — a strong,
   pre-stated over-determination test. *(Resolved at bridge grade by the §7 addendum; the NATIVE
   derivation remains open.)*
2. **The native phase space** (companion §6.1): 1/(48π) is Type-3; the Clifford γ-trace route is
   the open derivation; band-geometric route closed by computation.
3. Stated-not-applied small terms: QED FSR (+0.02%), m_b/m_τ/m_c phase space (−0.05%), QCD 3rd
   order (−0.01%) — none can absorb +0.44%; listed to keep the accounting honest, deliberately
   not applied to avoid assembly-tuning. *(Now bundled inside the registered layer — §7.)*

---

## 7. ADDENDUM (2026-07-02, user gate) — the EW layer REGISTERED via the LOOP program

The open mechanism of §6.1 was resolved at **bridge grade** by the loop program, and the layer
registered into the assembly (single source: `ew_width_layer.py`; full derivation in
`ew_width_layer_derivation.md`). The chain, every step pre-registered and git-witnessed:

1. **Class selection** (C2, pre-reg 2188fbe): the CAR-KMS matter loop on the P3 vertex forms is
   the FIRST O(1)-coefficient class for the demand; conditional on the P3/PS identification its
   content is standard EW ⟹ SM-REPRODUCTION-CONDITIONAL.
2. **Evaluation rule derived** (V1, pre-reg a5287f4): the KMS loop family has exactly two
   parameter-free evaluations; the arrow (the already-counted bit) selects the retarded VACUUM
   loop; thermality enters as statistics only. Machinery calibrated to the S2a standard
   (Veltman Δρ symbolic-exact; the 1/(12π) massless lock at 1e-14).
3. **Blind evaluation** (V2, pre-reg d37a679): δ_Z = −0.4864% = −1.81 loop units vs the
   pre-registered demand −1.62 ± 0.34 — **pull −0.54, LANDING** under the pre-registered tier
   rule; all falsification surfaces held (Γ_W/Γ_Z sub-σ; poles untouched; Γ_e = 0).

**Updated result:** Γ_Z/M_Z = 0.027483 × (1 + δ_Z) = **0.027350**; deviation **−0.55σ_PDG =
Clause 8c PASS** — numerically equal to the SM's own −0.53σ residual on this observable, which
is the honest content of SM-reproduction grade (the framework closes TO the SM, not to zero).

**Grade after registration:** STRUCTURAL-DERIVATION-CONDITIONAL, Clause 9(9b) bridge tag
explicit — the layer's numerical content is continuum-loop (π-transcendental over K);
K-rationality is BROKEN and acknowledged; the row can never pass bridge-conditional until the
native derivation (the interacting sector coupling / walk↔Fock dictionary at theorem grade,
todo §7) lands. The M_Z pole oblique residual is NOT touched (the layer dresses rates only —
the R3 rate clause), so the "two observables on one frontier" over-determination of §6.1
resolves as: the width observable closed at bridge grade; the pole observable stays open on the
static-dressing side.
