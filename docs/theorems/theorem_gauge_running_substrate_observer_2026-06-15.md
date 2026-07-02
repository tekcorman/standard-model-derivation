# Theorem (conditional): the electroweak sector = substrate boundary + observer running

**Date:** 2026-06-15
**Grade.** Three results are solid and standalone: (S1) the substrate boundary values, (S2)
the substrate/observer separation that dissolves the "24-vs-grade-lock" tension, (S3) the
unification-scale over-determination. One result is exact arithmetic **conditional** on the
mechanism: (C) the non-supersymmetric reproduction of the MSSM β-coefficients from the
observer's four-dimensional completion. One ingredient is **open** (the deep frontier): (O)
that the four-dimensional Dirac operator D₄ geometrically produces that completion.

Companion (untracked working material): `docs/scoping/ew_sector_arc_closure_2026-06-15.md`,
`docs/scoping/observer_time_dirac_dN_specification_2026-06-14.md`,
`docs/scoping/deep_weave_ncg_pati_salam_2026-06-14.md`; probes `proofs/_scratch/march*`.

---

## Statement

The framework's gauge sector separates cleanly into a **static substrate** part and a
**dynamical observer** part — the framework's own substrate-is-static / observer-owns-the-
dynamics distinction — and this separation resolves the long-standing tension over whether
the substrate's unified coupling α_GUT⁻¹ = 24 can reproduce the observed electroweak
couplings without supersymmetry.

**Substrate (static).** The substrate fixes, as running-independent structural numbers:
- the unified coupling **α_GUT⁻¹ = 2^{k\*}·k\* = 24**,
- the weak mixing **sin²θ_W = k\*/2^{k\*} = 3/8** at unification (with sin²θ_W·α_GUT⁻¹ = k\*² = 9),
- the unification scale **Λ = (2/3⁹)·M_Pl ≈ 1.24×10¹⁵ GeV**,
- the gauge group **SU(4)×SU(2)_L×SU(2)_R = Spin(6) = Cl(6)** (Pati–Salam), and
- two-Higgs-doublet field content; the gaugino field is **grade-excluded** (the adjoint **15**
  occupies only the even Clifford grades of Cl(6)).

The framework is thus a **discrete realization of the noncommutative-geometry spectral
Pati–Salam triple** (Chamseddine–Connes–van Suijlekom, arXiv:1304.8050, 1507.08161), a
*non-supersymmetric* unification whose boundary values are exactly the substrate's structural
numbers.

**Observer (the N-flow = the fourth/time direction).** The gauge β-functions are the
*running*, i.e. d/d ln N along the observer's event-count direction τ = log N — an observer
quantity, not substrate content.

**(S2) The split dissolves the tension.** "α_GUT⁻¹ = 24 reproduces the data only with the
MSSM β-coefficients, whose +4 superpartner content the grade-lock forbids" conflates two
objects. The grade-lock constrains the substrate's static **fields**; the "+4" is the
observer's **running**. They are different objects, and the grade-lock places no constraint
on the running. A non-supersymmetric substrate is therefore *compatible* with MSSM-shaped
β-coefficients.

**(C) The observer's four-dimensional completion supplies the "+4" non-supersymmetrically.**
Completing the substrate's three-dimensional Weyl spinors to four-dimensional Dirac spinors
adds, to each field, its **partner-equivalent** (the time-component flips statistics):
sfermion-equivalent (scalar), higgsino-equivalent (fermion), and gaugino-equivalent (adjoint
fermion, **non-abelian only**, since the U(1) "gaugino" — the bino — is a singlet). With three
generations (Σ T_f = 6 per group) this gives the MSSM β-coefficients **exactly**:

| coupling | sfermion-eq | higgsino-eq | gaugino-eq | b (4D) | MSSM |
|---|---|---|---|---|---|
| b₁ (U(1)) | 2 | 2/5 | **0** | 33/5 | 33/5 |
| b₂ (SU(2)) | 2 | 2/3 | 4/3 | 1 | 1 |
| b₃ (SU(3)) | 2 | 0 | 2 | −3 | −3 |

with **b₁ = 12/5 fixed structurally** (no free choice: the U(1) adjoint is trivial). Hence
α_GUT⁻¹ = 24 runs to the observed electroweak couplings — **1/α_EM = 127, sin²θ_W = 0.230,
α_s = 0.122** — non-supersymmetrically, with no adopted inputs.

**Falsifiable signature.** The partner-equivalents are four-dimensional time-component
*shadows* of the existing fields: they contribute to the **running** but do not propagate as
particles. Prediction: **MSSM-running with no physical superpartners** — distinct from the
MSSM (a sparticle spectrum) and from the SM/2HDM (no unification).

---

## Solid, standalone (S1–S3)

- **(S1)** α_GUT⁻¹ = 24, sin²θ_W = 3/8 structural, running-independent; gaugino grade-excluded
  (theorem, companion work on the Cl(6) grade structure).
- **(S2)** Substrate/observer separation: the gauge β is an observer-flow quantity, so the
  grade-lock (substrate content) does not constrain it. The "24-vs-grade-lock wall" was a
  category error.
- **(S3) Scale over-determination.** Λ = (2/3⁹)·M_Pl is derived in the **neutrino** sector
  (the ν_R Majorana scale, with y_ν = y_top forced by the (4,2,1) multiplet, giving
  m_ν₃ ≈ 50 meV). It equals the **gauge** unification scale M_GUT to ~1% — a genuine
  cross-sector over-determination, independent of the coupling value.

## Conditional bridge (C)

The partner-equivalent content reproduces the MSSM β-coefficients exactly. On the content
side this is "MSSM-content → MSSM-betas"; the non-trivial content is the **uniqueness** (it is
the unique non-supersymmetric content giving all three) and the **structural** passing of b₁
(gaugino-eq adjoint and non-abelian-only ⇒ the bino contributes nothing to b₁).

## Open: the mechanism (O)

That the four-dimensional Dirac operator **D₄ = D₃ ⊗ 1 + γ_t ⊗ ∂_N** geometrically produces
exactly these partner-equivalents (the KO-dimension 2 → 6 doubling), i.e. that ζ_{D₄}(0) = the
+4. The obstruction is making the **irreversible** observer-time direction ∂_N (the arrow /
the entropic N-growth) into a self-adjoint Dirac direction — likely a twisted (dilation-by-N)
or Lorentzian spectral triple. Pre-registered kill criterion: if ∂_N cannot be made a
consistent Dirac/time direction, there is no D₄ and (C) stays conditional; α_GUT⁻¹ = 24
remains a structural number whose realized-unification reading is conditional on the running.

---

## One line

The gauge sector is substrate-static (24, 3/8, the scale, 2HDM, grade-locked) plus
observer-running; the observer's four-dimensional completion supplies the MSSM β-coefficients
non-supersymmetrically (b₁ structural), so 24 runs to the observed electroweak sector — with a
falsifiable no-superpartner signature — conditional only on the D₄ construction.
